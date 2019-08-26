# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn
from scipy.stats import norm, sem
from scipy.stats.t import ppf
from sklearn.utils import resample
from .utils import restore_segmentation
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from src.data.loader import load_custom_data
logger = getLogger()


TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH


class EvaluatorMT(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.discriminator = trainer.discriminator
        self.data = data
        self.dico = data['dico']
        self.params = params

        # create reference files for BLEU evaluation
        self.create_reference_files()

    def get_pair_for_mono(self, lang):
        """
        Find a language pair for monolingual data.
        """
        candidates = [(l1, l2) for (l1, l2) in self.data['para'].keys() if l1 == lang or l2 == lang]
        assert len(candidates) > 0
        return sorted(candidates)[0]

    def mono_iterator(self, data_type, lang):
        """
        If we do not have monolingual validation / test sets, we take one from parallel data.
        """
        dataset = self.data['mono'][lang][data_type]
        if dataset is None:
            pair = self.get_pair_for_mono(lang)
            dataset = self.data['para'][pair][data_type]
            i = 0 if pair[0] == lang else 1
        else:
            i = None
        dataset.batch_size = self.params.batch_size
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch if i is None else batch[i]

    def get_iterator(self, data_type, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['valid', 'test']
        if lang2 is None or lang1 == lang2:
            for batch in self.mono_iterator(data_type, lang1):
                yield batch if lang2 is None else (batch, batch)
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k][data_type]
            dataset.batch_size = self.params.batch_size
            for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
                yield batch if lang1 < lang2 else batch[::-1]

    def get_paraphrase_iterator(self, data_type, lang):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['test_real', 'test_fake']

        dataset = self.data['paraphrase'][lang][data_type]
        dataset.batch_size = self.params.batch_size
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2
            lang1_id = params.lang2id[lang1]
            lang2_id = params.lang2id[lang2]

            for data_type in ['valid', 'test']:

                lang1_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_type))
                lang2_path = os.path.join(params.dump_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_type))

                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_type, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico[lang1], lang1_id, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico[lang2], lang2_id, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # restore original segmentation

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    lang1_txt = '\n'.join(lang1_txt) + '\n'
                    lang1_txt = restore_segmentation(lang1_txt)
                    f.write(lang1_txt)
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    lang2_txt = '\n'.join(lang2_txt) + '\n'
                    lang2_txt = restore_segmentation(lang2_txt)
                    f.write(lang2_txt)

                # store data paths
                params.ref_paths[(lang2, lang1, data_type)] = lang1_path
                params.ref_paths[(lang1, lang2, data_type)] = lang2_path

    def eval_para(self, lang1, lang2, data_type, scores):
        """
        Evaluate lang1 -> lang2 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn2 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang2_id].weight, size_average=False)
        n_words2 = self.params.n_words[lang2_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang2):

            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.cuda(), sent2.cuda()

            # encode / decode / generate
            encoded = self.encoder(sent1, len1, lang1_id, noise=0)
            decoded = self.decoder(encoded, sent2[:-1], lang2_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # cross-entropy loss
            xe_loss += loss_fn2(decoded.view(-1, n_words2), sent2[1:].view(-1)).item()
            count += (len2 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        ref_path = params.ref_paths[(lang1, lang2, data_type)]

        
        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            txt = '\n'.join(txt) + '\n'
            txt = restore_segmentation(txt)            
            f.write(txt)

        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

        # update scores
        scores['ppl_%s_%s_%s' % (lang1, lang2, data_type)] = np.exp(xe_loss / count)
        scores['bleu_%s_%s_%s' % (lang1, lang2, data_type)] = bleu

    def eval_back(self, lang1, lang2, lang3, data_type, scores):
        """
        Compute lang1 -> lang2 -> lang3 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s -> %s (%s) ..." % (lang1, lang2, lang3, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn3 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang3_id].weight, size_average=False)
        n_words3 = self.params.n_words[lang3_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang3):

            # batch
            (sent1, len1), (sent3, len3) = batch
            sent1, sent3 = sent1.cuda(), sent3.cuda()

            # encode / generate lang1 -> lang2
            encoded = self.encoder(sent1, len1, lang1_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # encode / decode / generate lang2 -> lang3
            encoded = self.encoder(sent2_.cuda(), len2_, lang2_id)
            decoded = self.decoder(encoded, sent3[:-1], lang3_id)
            sent3_, len3_, _ = self.decoder.generate(encoded, lang3_id)

            # cross-entropy loss
            xe_loss += loss_fn3(decoded.view(-1, n_words3), sent3[1:].view(-1)).item()
            count += (len3 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent3_, len3_, self.dico[lang3], lang3_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}-{3}.{4}.txt'.format(scores['epoch'], lang1, lang2, lang3, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        if lang1 == lang3:
            _lang1, _lang3 = self.get_pair_for_mono(lang1)
            if lang3 != _lang3:
                _lang1, _lang3 = _lang3, _lang1
            ref_path = params.ref_paths[(_lang1, _lang3, data_type)]
        else:
            ref_path = params.ref_paths[(lang1, lang3, data_type)]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            txt = '\n'.join(txt) + '\n'
            txt = restore_segmentation(txt)            
            f.write(txt)

        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

        # update scores
        scores['ppl_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = np.exp(xe_loss / count)
        scores['bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = bleu

    def eval_translation_recog(self, lang1, lang2, data_type, scores):
        """
        Evaluate translation recognition, which involves
        1. Computing cosine similarities on src ref to tgt ref
        2. Computing cosine similarities on src ref to wrong tgt ref (i.e. fake ref)
        3. Computing a decision criterion and separability score
        """
        logger.info("Evaluating %s -> %s translation recognition (%s) ..." % (lang1, lang2, data_type))
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        real_sims = torch.Tensor().cuda()
        fake_sims = torch.Tensor().cuda()
        for batch in self.get_iterator(data_type, lang1, lang2):

            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.cuda(), sent2.cuda()

            # encode / decode / generate
            encoded1 = self.encoder(sent1, len1, lang1_id, noise=0)
            encoded2 = self.encoder(sent2, len2, lang2_id, noise=0)

            real_ref = encoded2.dis_input
            fake_ref = torch.cat((real_ref[:, 1:], real_ref[:, 0].unsqueeze(1)), dim=1)

            real_sim = cosine_sim(encoded1.dis_input, real_ref)
            fake_sim = cosine_sim(encoded1.dis_input, fake_ref)
            real_sims = torch.cat((real_sims, real_sim))
            fake_sims = torch.cat((fake_sims, fake_sim))

        real_mean_cos_sim = real_sims.mean(dim=0).item()
        real_std_cos_sim = real_sims.std(dim=0).item()
        logger.info("REAL_MEAN_COS_SIM : %f" % (real_mean_cos_sim))
        logger.info("REAL_STD_COS_SIM : %f" % (real_std_cos_sim))
        fake_mean_cos_sim = fake_sims.mean(dim=0).item()
        fake_std_cos_sim = fake_sims.std(dim=0).item()
        logger.info("FAKE_MEAN_COS_SIM : %f" % (fake_mean_cos_sim))
        logger.info("FAKE_STD_COS_SIM : %f" % (fake_std_cos_sim))

        if real_mean_cos_sim < fake_mean_cos_sim:
            sc = sim_score(real_mean_cos_sim, real_std_cos_sim, fake_mean_cos_sim, fake_std_cos_sim)
        else:
            sc = sim_score(fake_mean_cos_sim, fake_std_cos_sim, real_mean_cos_sim, real_std_cos_sim)

        sc2 = real_sim_score(fake_sims, real_sims)

        logger.info("T_SIM_SCORE: %f" % (sc))
        logger.info("RT_SIM_SCORE: %f" % (sc2))


        logger.info("Bootstrap-resampled:")
        scs = np.zeros(100)
        for i in range(100):
            fake_sample, real_sample = resample(fake_sims, real_sims, n_samples=300)
            scs[i] = real_sim_score(fake_sample, real_sample)


        h = sem(scs) * ppf((1 + 0.95) / 2.0, len(scs)-1)
        logger.info("Resampled RT_SIM_SCORE: %f +- %f" % (np.mean(scs), h))
        logger.info("Resampled RT_SIM_SCORE stdev: %f" % (np.std(scs)))

        # update scores
        # scores['meancossim_%s_%s' % (lang, 'real')] = real_mean_cos_sim
        # scores['stdcossim_%s_%s' % (lang, 'real')] = real_std_cos_sim

        # if scores['meancossim_%s_%s' % (lang, 'test_real')] < scores['meancossim_%s_%s' % (lang, 'test_fake')]:
        #     # this really should never happen except for maybe early in training...
        #     sc = sim_score(
        #         scores['meancossim_%s_%s' % (lang, 'test_fake')],
        #         scores['stdcossim_%s_%s' % (lang, 'test_fake')],
        #         scores['meancossim_%s_%s' % (lang, 'test_real')],
        #         scores['stdcossim_%s_%s' % (lang, 'test_real')])
        # else:
        #     sc = sim_score(
        #         scores['meancossim_%s_%s' % (lang, 'test_fake')],
        #         scores['stdcossim_%s_%s' % (lang, 'test_fake')],
        #         scores['meancossim_%s_%s' % (lang, 'test_real')],
        #         scores['stdcossim_%s_%s' % (lang, 'test_real')])

        #     logger.info("SIM_SCORE: %f" % (sc))

        # # update scores
        # scores['simscore_%s' % (lang)] = sc
    def eval_paraphrase_recog(self, lang, scores):
        """
        Evaluate lang paraphrase recognition, which involves
        1. Computing cosine similarities on real paraphrases
        2. Computing cosine similarities on fake paraphrases
        3. Computing a decision criterion and separability score
        """
        logger.info("Evaluating %s paraphrase ..." % (lang))
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang_id = params.lang2id[lang]

        # hypothesis
        txt = []

        real_sims = torch.Tensor().cuda()
        fake_sims = torch.Tensor().cuda()
        for batch in self.get_paraphrase_iterator('test_real', lang):
            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.cuda(), sent2.cuda()

            # encode / decode / generate
            encoded1 = self.encoder(sent1, len1, lang_id, noise=0)
            encoded2 = self.encoder(sent2, len2, lang_id, noise=0)

            real_ref = encoded2.dis_input
            fake_ref = torch.cat((real_ref[:, 1:], real_ref[:, 0].unsqueeze(1)), dim=1)

            real_sim = cosine_sim(encoded1.dis_input, real_ref)
            fake_sim = cosine_sim(encoded1.dis_input, fake_ref)
            real_sims = torch.cat((real_sims, real_sim))
            fake_sims = torch.cat((fake_sims, fake_sim))

        real_mean_cos_sim = real_sims.mean(dim=0).item()
        real_std_cos_sim = real_sims.std(dim=0).item()
        logger.info("REAL_MEAN_COS_SIM : %f" % (real_mean_cos_sim))
        logger.info("REAL_STD_COS_SIM : %f" % (real_std_cos_sim))
        fake_mean_cos_sim = fake_sims.mean(dim=0).item()
        fake_std_cos_sim = fake_sims.std(dim=0).item()
        logger.info("FAKE_MEAN_COS_SIM : %f" % (fake_mean_cos_sim))
        logger.info("FAKE_STD_COS_SIM : %f" % (fake_std_cos_sim))

        if real_mean_cos_sim < fake_mean_cos_sim:
            sc = sim_score(real_mean_cos_sim, real_std_cos_sim, fake_mean_cos_sim, fake_std_cos_sim)
        else:
            sc = sim_score(fake_mean_cos_sim, fake_std_cos_sim, real_mean_cos_sim, real_std_cos_sim)

        sc2 = real_sim_score(fake_sims, real_sims)

        logger.info("P_SIM_SCORE: %f" % (sc))
        logger.info("RP_SIM_SCORE: %f" % (sc2))


        logger.info("Bootstrap-resampled:")
        scs = np.zeros(100)
        for i in range(100):
            fake_sample, real_sample = resample(fake_sims, real_sims, n_samples=300)
            scs[i] = real_sim_score(fake_sample, real_sample)


        h = sem(scs) * ppf((1 + 0.95) / 2.0, len(scs)-1)
        logger.info("Resampled RP_SIM_SCORE: %f +- %f" % (np.mean(scs), h))
        logger.info("Resampled RP_SIM_SCORE stdev: %f" % (np.std(scs)))

    # def eval_paraphrase_recog(self, lang, scores):
    #     """
    #     Evaluate lang paraphrase recognition, which involves
    #     1. Computing cosine similarities on real paraphrases
    #     2. Computing cosine similarities on fake paraphrases
    #     3. Computing a decision criterion and separability score
    #     """
    #     for data_type in ['test_real', 'test_fake']:
    #         logger.info("Evaluating %s paraphrase (%s) ..." % (lang, data_type))
    #         self.encoder.eval()
    #         self.decoder.eval()
    #         params = self.params
    #         lang_id = params.lang2id[lang]

    #         # hypothesis
    #         txt = []

    #         all_sim = 0 
    #         n_sents = 0
    #         sims = torch.Tensor().cuda()
    #         for batch in self.get_paraphrase_iterator(data_type, lang):

    #             # batch
    #             (sent1, len1), (sent2, len2) = batch
    #             sent1, sent2 = sent1.cuda(), sent2.cuda()

    #             # encode / decode / generate
    #             encoded1 = self.encoder(sent1, len1, lang_id, noise=0)
    #             encoded2 = self.encoder(sent2, len2, lang_id, noise=0)

    #             sim = cosine_sim(encoded1.dis_input, encoded2.dis_input)
    #             sims = torch.cat((sims, sim))

    #         mean_cos_sim = sims.mean(dim=0).item()
    #         std_cos_sim = sims.std(dim=0).item()
    #         logger.info("MEAN_COS_SIM : %f" % (mean_cos_sim))
    #         logger.info("STD_COS_SIM : %f" % (std_cos_sim))

    #         # update scores
    #         scores['meancossim_%s_%s' % (lang, data_type)] = mean_cos_sim
    #         scores['stdcossim_%s_%s' % (lang, data_type)] = std_cos_sim

    #     if scores['meancossim_%s_%s' % (lang, 'test_real')] < scores['meancossim_%s_%s' % (lang, 'test_fake')]:
    #         # this really should never happen except for maybe early in training...
    #         sc = sim_score(
    #             scores['meancossim_%s_%s' % (lang, 'test_fake')],
    #             scores['stdcossim_%s_%s' % (lang, 'test_fake')],
    #             scores['meancossim_%s_%s' % (lang, 'test_real')],
    #             scores['stdcossim_%s_%s' % (lang, 'test_real')])
    #     else:
    #         sc = sim_score(
    #             scores['meancossim_%s_%s' % (lang, 'test_fake')],
    #             scores['stdcossim_%s_%s' % (lang, 'test_fake')],
    #             scores['meancossim_%s_%s' % (lang, 'test_real')],
    #             scores['stdcossim_%s_%s' % (lang, 'test_real')])

    #     logger.info("SIM_SCORE: %f" % (sc))

    #     # update scores
    #     scores['simscore_%s' % (lang)] = sc

    def eval_paraphrase_gen(self, lang, scores):
        """
        Evaluate lang - lang paraphrases
        1. Cosine similarity
        2. BLEU/PINC 
        3. METEOR/PINC
        4. SES/PINC
        """
        for data_type in ['test_real', 'test_fake']:
            logger.info("Evaluating %s paraphrase (%s) ..." % (lang, data_type))
            self.encoder.eval()
            self.decoder.eval()
            params = self.params
            lang_id = params.lang2id[lang]

            # hypothesis
            txt = []

            all_sim = 0 
            n_sents = 0
            sims = torch.Tensor()
            for batch in self.get_paraphrase_iterator(data_type, lang):

                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                encoded1 = self.encoder(sent1, len1, lang_id, noise=0)

                sim = cosine_sim(encoded1.dis_input, encoded2.dis_input)
                sims = torch.cat((sims, sim))
                # n_sents += sim.size(0)
                # all_sim += sim.sum(dim=0)


                # decoded = self.decoder(encoded, sent2[:-1], lang2_id)
                # sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

                # convert to text
                # txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

            # hypothesis / reference paths
            # hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type)
            # hyp_path = os.path.join(params.dump_path, hyp_name)
            # ref_path = params.ref_paths[(lang1, lang2, data_type)]

            
            # export sentences to hypothesis file / restore BPE segmentation
            # with open(hyp_path, 'w', encoding='utf-8') as f:
            #     txt = '\n'.join(txt) + '\n'
            #     txt = restore_segmentation(txt)            
            #     f.write(txt)

            # evaluate BLEU score
            # bleu = eval_moses_bleu(ref_path, hyp_path)
            # logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            mean_cos_sim = sims.mean(dim=0).item()
            std_cos_sim = sims.std(dim=0).item()
            logger.info("MEAN_COS_SIM : %f" % (mean_cos_sim))
            logger.info("STD_COS_SIM : %f" % (std_cos_sim))

            # update scores
            scores['meancossim_%s_%s' % (lang, data_type)] = mean_cos_sim.item()
            scores['stdcossim_%s_%s' % (lang, data_type)] = std_cos_sim.item()

    def custom_eval(self, filepath, input_lang, imd_lang, output_lang, scores):
        """
        Run on custom data with custom methods
        """
        dataset = load_custom_data(filepath, input_lang, self.params, self.data)
        input_lang_id = self.params.lang2id[input_lang]
        if imd_lang is not None:
           imd_lang_id = self.params.lang2id[imd_lang]
        output_lang_id = self.params.lang2id[output_lang]
        self.encoder.eval()
        self.decoder.eval()


        for i in range(10):
            txt = []
            for batch in dataset.get_iterator(shuffle=False)():
                (sent1, len1) = batch
                sent1 = sent1.cuda()
                
                if imd_lang is None:
                    encoded = self.encoder(sent1, len1, input_lang_id, noise=i*1.0)
                    sent3_, len3_, _ = self.decoder.generate(encoded, output_lang_id)

                else:
                    encoded = self.encoder(sent1, len1, input_lang_id, noise=0)
                    sent2_, len2_, _ = self.decoder.generate(encoded, imd_lang_id)

                    # encode / decode / generate lang2 -> lang3
                    encoded = self.encoder(sent2_.cuda(), len2_, imd_lang_id, noise=i*1.0)
                    sent3_, len3_, _ = self.decoder.generate(encoded, output_lang_id)


                # txt.extend(convert_to_text(sent2_, len2_, self.dico[output_lang], output_lang_id, self.params))
                txt.extend(convert_to_text(sent3_, len3_, self.dico[output_lang], output_lang_id, self.params))

            # hypothesis / reference paths
            if imd_lang is None:
                hyp_name = 'cust{0}.{1}-{2}.txt'.format(i, input_lang, output_lang)
            else: 
                hyp_name = 'cust{0}.{1}-{2}--{3}.txt'.format(i, input_lang, imd_lang, output_lang)
            hyp_path = os.path.join(self.params.dump_path, hyp_name)

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                txt = '\n'.join(txt) + '\n'
                txt = restore_segmentation(txt)            
                f.write(txt)


            # encode / decode / generate

    def run_vae_evals(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():

            for lang1, lang2 in self.data['para'].keys():
                self.translate_vae(lang1, lang2, 'test', scores)
                self.translate_vae(lang2, lang1, 'test', scores)                
                self.paraphrase_vae(lang1, lang2, 'test', scores)
                self.paraphrase_vae(lang2, lang1, 'test', scores)   
        return scores


    def run_all_evals(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():
            if self.params.eval_only:
                filepath = "data/pp/coco/captions_val2014.src-filtered.en.tok.60000.pth"
                # self.custom_eval(filepath, 'en', None, 'en', scores)
                self.custom_eval(filepath, 'en','fr', 'en', scores)


            for lang in self.data['paraphrase'].keys():
            #     # print('LANG LANG', lang)
            #     # self.multi_sample_eval(lang, lang, 'test_real', scores)
                self.eval_paraphrase_recog(lang, scores)

            for lang1, lang2 in self.data['para'].keys():
                for data_type in ['valid', 'test']:
                    self.eval_translation_recog(lang1, lang2, data_type, scores)
                    self.eval_translation_recog(lang2, lang1, data_type, scores)
                    # self.eval_translation_recog(lang2, lang1, data_type, scores)
                    #if self.params.eval_only and self.params.variational:
                    #     self.variation_eval(lang1, lang2, data_type, scores)
                    #    self.multi_sample_eval(lang1, lang2, data_type, scores)
                    #    self.multi_sample_eval(lang2, lang1, data_type, scores)

                    
                    self.eval_para(lang1, lang2, data_type, scores)
                    self.eval_para(lang2, lang1, data_type, scores)

                    

            for lang1, lang2, lang3 in self.params.pivo_directions:
                for data_type in ['valid', 'test']:
                    self.eval_back(lang1, lang2, lang3, data_type, scores)


        return scores

    def translate_vae(self, lang1, lang2, data_type, scores):
        """
            Samples several times and 
        """
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
    
        subprocess.Popen("mkdir -p %s" % os.path.join(params.dump_path, 'vae'), shell=True).wait()
        # hypothesis
        for i in range(params.eval_samples):
            txt = []
            logger.info("i = %d" % i)
            for j, batch in enumerate(self.get_iterator(data_type, lang1, lang2)):
            
                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                # if i == 0: # first eval always the "most likely output"
                #     encoded = self.encoder(sent1, len1, lang1_id, noise=0)
                # else:
                encoded = self.encoder(sent1, len1, lang1_id, noise=5.0*i)
                sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

                txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

                if j == 10:
                    break
            # break
        
            hyp_name = 'vae-sample{4}-epoch{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type, i)
            hyp_path = os.path.join(params.dump_path, 'vae', hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_type)]
            
            with open(hyp_path, 'w', encoding='utf-8') as f:
                txt = '\n'.join(txt) + '\n'
                txt = restore_segmentation(txt)            
                f.write(txt)
            
        # restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s"
        # os.system(restore_cmd % hyp_path)

    def multi_sample_eval(self, lang1, lang2, data_type, scores):
        """
            Samples several times and chooses with a choice function
        """
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
    
        subprocess.Popen("mkdir -p %s" % os.path.join(params.dump_path, 'vae'), shell=True).wait()
        # hypothesis

        iterator = lambda: self.get_paraphrase_iterator(data_type, lang1) \
            if data_type in ['test_real', 'test_fake'] \
            else self.get_iterator(data_type, lang1, lang2)


        src_txt = []
        tgt_txt = []
        hyp_txts = []
        for i in range(params.eval_samples):
            txt = []
            # logger.info("i = %d" % i)
            for j, batch in enumerate(iterator()):
            
                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                if i == 0: # first eval always the "most likely output"
                    encoded = self.encoder(sent1, len1, lang1_id, noise=0)
                else:
                    encoded = self.encoder(sent1, len1, lang1_id, noise=1.0*i)
                sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id, max_len=5*max(len1))

                txt.extend(restore_segmentation(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params)))
                if i == 0:
                    src_txt.extend(restore_segmentation(convert_to_text(sent1, len1, self.dico[lang1], lang1_id, self.params)))
                    tgt_txt.extend(restore_segmentation(convert_to_text(sent2, len2, self.dico[lang2], lang2_id, self.params)))

            hyp_txts.append(txt)

        for i, hyp_txt in enumerate(hyp_txts):
            bleui = eval_nltk_bleu(tgt_txt, hyp_txt)
            logger.info("BLEU #%d: %f" % (i, bleui))
            hyp_path = os.path.join(params.dump_path, 'multic{0}.{1}-{2}.txt'.format(i, lang1, lang2))
            with open(hyp_path,'w', encoding='utf-8') as f:
                f.write('\n'.join(hyp_txt)+'\n')

        final_hyp_txt = choose_sentences(hyp_txts, 'best', self.params, tgt_txt=tgt_txt)

        final_bleu_score = eval_nltk_bleu(tgt_txt, final_hyp_txt)*100
        final_pinc_score = eval_pinc(src_txt, final_hyp_txt)


        logger.info("MULTI_BLEU : %f" % (final_bleu_score))
        logger.info("MULTI_PINC : %f" % (final_pinc_score))

            # update scores
        scores['multi_bleu_%s_%s_%s' % (lang1, lang2, data_type)] = final_bleu_score
        scores['multi_pinc_%s_%s_%s' % (lang1, lang2, data_type)] = final_pinc_score


    def variation_eval(self, lang1, lang2, data_type, scores):
        """
            Measure variation of a VAE by checking bleu scores between sentences
        """
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
    
        subprocess.Popen("mkdir -p %s" % os.path.join(params.dump_path, 'vae'), shell=True).wait()
        # hypothesis

        iterator = lambda: self.get_paraphrase_iterator(data_type, lang1) \
            if data_type in ['test_real', 'test_fake'] \
            else self.get_iterator(data_type, lang1, lang2)

        hyp_txts = []
        for i in range(10):
            txt = []
            # logger.info("i = %d" % i)
            for j, batch in enumerate(iterator()):
            
                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                # if i == 0: # first eval always the "most likely output"
                #     encoded = self.encoder(sent1, len1, lang1_id, noise=0)
                # else:
                encoded = self.encoder(sent1, len1, lang1_id, noise=20.0*i)
                sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

                txt.extend(restore_segmentation(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params)))
                if j == 10:
                    break
            hyp_txts.append(txt)

        for i in range(1, 10):
            final_bleu_score = eval_nltk_bleu(hyp_txts[0], hyp_txts[i])*100
            logger.info("VAR_BLEU_%s_%s_%s (%d): %f" % (lang1, lang2, data_type, 10.0*i, final_bleu_score))
        # final_pinc_score = eval_pinc(src_txt, final_hyp_txt)

        # logger.info("MULTI_PINC : %f" % (final_pinc_score))

            # update scores
        # scores['multi_bleu_%s_%s_%s' % (lang1, lang2, data_type)] = final_bleu_score
        # scores['multi_pinc_%s_%s_%s' % (lang1, lang2, data_type)] = final_pinc_score


    def paraphrase_vae(self, lang1, lang2, data_type, scores):
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
    
        # hypothesis
        for i in range(10):
            txt = []
            logger.info("i = %d" % i)
            for j, batch in enumerate(self.get_iterator(data_type, lang1, lang2)):
            
                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                # if i == 0: # first eval always the "most likely output"
                #     encoded = self.encoder(sent1, len1, lang1_id, noise=0)
                # else:
                encoded = self.encoder(sent1, len1, lang1_id, noise=5.0*i)
                sent2_, len2_, _ = self.decoder.generate(encoded, lang1_id)

                txt.extend(convert_to_text(sent2_, len2_, self.dico[lang1], lang1_id, self.params))

                if j == 10:
                    break
            # break
        
            hyp_name = 'pvae--{4}--{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type, i)
            hyp_path = os.path.join(params.dump_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_type)]
            
            with open(hyp_path, 'w', encoding='utf-8') as f:
                txt = '\n'.join(txt) + '\n'
                txt = restore_segmentation(txt)            
                f.write(txt)
        

def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1

def eval_nltk_bleu(ref, hyp):
    """
    Given texts of structure: [ref1, ref2, ref3], [hyp1, hyp2, hyp3]
    Convert to proper structure for corpus_bleu, and run it.
    """
    ref_bleu = [[r.split()] for r in ref]
    hyp_bleu = [h.split() for h in hyp]
    return corpus_bleu(ref_bleu, hyp_bleu)

def eval_pinc(ref, hyp):
    pinc_sum = 0.0
    for i in range(len(ref)):
        pinc_sum += pinc_score(ref[i].split(), hyp[i].split())
    return pinc_sum / len(ref)

def pinc_score(src_sen, can_sen, max_ngram=4):
    """
    PINC score, as defined in 
    "Collecting Highly Parallel Data for Paraphrase Evaluation" (Chen et al. 2011)
    """
    src_ngrams = {}
    for n in range(max_ngram):
        src_ngrams[n] = [src_sen[i:i+n+1] for i in range(len(src_sen)-n)]

    ngram_counts = np.zeros(max_ngram)
    ngram_totals = np.array([max(len(can_sen) - i + 1, 1) for i in range(1, max_ngram+1)])

    for n in range(max_ngram):
        for i in range(ngram_totals[n]):
            ngram_counts[n] += can_sen[i:n+i+1] in src_ngrams[n]

    pinc = np.sum(1 - ngram_counts/ngram_totals) / max_ngram
    return pinc

def cosine_sim(enc1, enc2):
    """
    Compute cosine similarity between two encodings. Using method from Cer et al. 2018
    """
    assert enc1.size()[1:] == enc2.size()[1:]
    vec1 = enc1.sum(dim=0) / np.sqrt(enc1.size(0))
    vec2 = enc2.sum(dim=0) / np.sqrt(enc2.size(0))
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)

def sim_score(mu1, sigma1, mu2, sigma2):
    """
    Compute the similarity score, which is 1 minus the area under the intersection of 2 gaussians
    """
    assert mu1 < mu2, "mu1 should be smaller than mu2."
    if sigma1 == sigma2:
        intersect = (mu1 + mu2)/2
    else:
        var1 = sigma1**2
        var2 = sigma2**2
        var_diff = var1-var2
        rat = np.sqrt((mu1 - mu2)**2 + 2*var_diff*np.log(sigma1 / sigma2))
        numer = mu2 * var1 - sigma2 * (mu1 * sigma2 + sigma1 * rat)
        intersect = numer / var_diff

    if not (mu1 < intersect < mu2):
        logger.warn("Intersection does not lie between mu1 and mu2.")
    # assert mu1 < intersect < mu2, "intersection is not between mu1 and mu2."
    # print(mu1, intersect, mu2)


    return norm.cdf(intersect, mu1, sigma1) - norm.cdf(intersect, mu2, sigma2)

def real_sim_score(fake_sims, real_sims):
    if type(fake_sims) is torch.Tensor:
        mu1 = fake_sims.mean(dim=0).item()
        sigma1 = fake_sims.std(dim=0).item()
        mu2 = real_sims.mean(dim=0).item()
        sigma2 = real_sims.std(dim=0).item()
    else:
        mu1 = np.mean(fake_sims)
        sigma1 = np.std(fake_sims)
        mu2 = np.mean(real_sims)
        sigma2 = np.std(real_sims)


    if sigma1 == sigma2:
        intersect = (mu1 + mu2)/2
    else:
        var1 = sigma1**2
        var2 = sigma2**2
        var_diff = var1-var2
        rat = np.sqrt((mu1 - mu2)**2 + 2*var_diff*np.log(sigma1 / sigma2))
        numer = mu2 * var1 - sigma2 * (mu1 * sigma2 + sigma1 * rat)
        intersect = numer / var_diff

    correct = ((real_sims >= intersect).sum() + (fake_sims < intersect).sum()).item()
    total = float(len(real_sims) + len(fake_sims))
    
    return correct/total
    


def choose_sentences(texts, method, params, tgt_txt=None):
    """ Choose sentences """
    if method == 'random':
        final_text = []
        for i in range(len(texts[0])):
            idx = np.random.choice(params.eval_samples)
            final_text.append(texts[idx][i])
        return final_text
    elif method == 'best': # choosing best BLEU, just to show theoretical max
        final_text = []
        for i in range(len(texts[0])):
            best_bleu = 0
            best_bleu_idx = 0
            for j in range(params.eval_samples):
                b = sentence_bleu([tgt_txt[i].split()], texts[j][i].split())
                if b > best_bleu:
                    best_bleu = b
                    best_bleu_idx = j
            final_text.append(texts[best_bleu_idx][i])
        return final_text
    else:
        pass # TODO: AWD choice

def convert_to_text(batch, lengths, dico, lang_id, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences
