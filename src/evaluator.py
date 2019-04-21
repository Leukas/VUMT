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
from scipy.stats import norm
from .utils import restore_segmentation


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
        dataset.batch_size = 32
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
            dataset.batch_size = 32
            for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
                yield batch if lang1 < lang2 else batch[::-1]

    def get_paraphrase_iterator(self, data_type, lang, real=True):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['test_real', 'test_fake']

        dataset = self.data['paraphrase'][lang][data_type]
        dataset.batch_size = 32
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

    def eval_paraphrase_recog(self, lang, scores):
        """
        Evaluate lang paraphrase recognition, which involves
        1. Computing cosine similarities on real paraphrases
        2. Computing cosine similarities on fake paraphrases
        3. Computing a decision criterion and separability score
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
            sims = torch.Tensor().cuda()
            for batch in self.get_paraphrase_iterator(data_type, lang):

                # batch
                (sent1, len1), (sent2, len2) = batch
                sent1, sent2 = sent1.cuda(), sent2.cuda()

                # encode / decode / generate
                encoded1 = self.encoder(sent1, len1, lang_id, noise=0)
                encoded2 = self.encoder(sent2, len2, lang_id, noise=0)

                sim = cosine_sim(encoded1.dis_input, encoded2.dis_input)
                sims = torch.cat((sims, sim))

            mean_cos_sim = sims.mean(dim=0).item()
            std_cos_sim = sims.std(dim=0).item()
            logger.info("MEAN_COS_SIM : %f" % (mean_cos_sim))
            logger.info("STD_COS_SIM : %f" % (std_cos_sim))

            # update scores
            scores['meancossim_%s_%s' % (lang, data_type)] = mean_cos_sim
            scores['stdcossim_%s_%s' % (lang, data_type)] = std_cos_sim

        if scores['meancossim_%s_%s' % (lang, 'test_real')] < scores['meancossim_%s_%s' % (lang, 'test_fake')]:
            # this really should never happen except for maybe early in training...
            sc = sim_score(
                scores['meancossim_%s_%s' % (lang, 'test_fake')],
                scores['stdcossim_%s_%s' % (lang, 'test_fake')],
                scores['meancossim_%s_%s' % (lang, 'test_real')],
                scores['stdcossim_%s_%s' % (lang, 'test_real')])
        else:
            sc = sim_score(
                scores['meancossim_%s_%s' % (lang, 'test_fake')],
                scores['stdcossim_%s_%s' % (lang, 'test_fake')],
                scores['meancossim_%s_%s' % (lang, 'test_real')],
                scores['stdcossim_%s_%s' % (lang, 'test_real')])

        logger.info("SIM_SCORE: %f" % (sc))

        # update scores
        scores['simscore_%s_%s' % (lang, data_type)] = sc

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
                encoded2 = self.encoder(sent2, len2, lang_id, noise=0)

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

            for lang in self.data['paraphrase'].keys():
                self.eval_paraphrase_recog(lang, scores)

            for lang1, lang2 in self.data['para'].keys():
                for data_type in ['valid', 'test']:
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
