import os
os.environ["LASER"]="%s/u2/tools/LASER/" % os.environ['HOME']
import subprocess
import numpy as np
import scipy.spatial.distance as dist
import argparse
# import source.embed as embed

# ref_folder = '%s/u2/metrics/wmt14-data/txt/references/' % os.environ['HOME']
# hyp_folder = '%s/u2/metrics/wmt14-data/txt/system-outputs/newstest2014/' % os.environ['HOME']

ref_folder = '%s/u2/metrics/wmt18-submitted-data/txt/references/' % os.environ['HOME']
hyp_folder = '%s/u2/metrics/wmt18-submitted-data/txt/system-outputs/newstest2018/' % os.environ['HOME']
output_folder = '%s/u2/metrics/encodings/wmt18/' % os.environ['HOME']


def embed(input_file, lang, output_file):
    ENCODER = os.environ['LASER']+'models/bilstm.93langs.2018-12-26.pt'
    BPE_CODES = os.environ['LASER']+'models/93langs.fcodes'
    EMBEDDER = 'python3 '+os.environ['LASER']+'source/embed.py '
    command = EMBEDDER + '< %s --encoder %s --token-lang %s --bpe-codes %s --output %s --verbose'
    p = subprocess.call(command % (input_file, ENCODER, lang, BPE_CODES, output_file), shell=True)
    # result = p.communicate()[0].decode("utf-8")
    # print(result)


def encode_refs(ref_files):
    print('Encoding reference files...')
    # parse and encode references
    for filename in ref_files: # something like: newstest2014-csen-ref.cs
        langs = filename.split("-")[1] # csen
        ref_lang = filename.split(".")[1] # cs

        if ref_lang == langs[2:]: # (en-cs if ref is cs) or (cs-en if ref is en)
            langs = langs[:2] + '-' + langs[2:] 
        else:
            langs = langs[2:] + '-' + langs[:2]

        # todo: tokenize ref file
        # tokenize(os.path.join(ref_folder, filename))
        input_file = os.path.join(ref_folder, filename)
        output_file = os.path.join(output_folder, 'references', '%s.%s' % (langs, ref_lang))
        embed(input_file, ref_lang, output_file)


def encode_hyps(hyp_files):
    print('Encoding hypothesis files...')
    # parse and encode system-outputs
    for lang_pair in hyp_files: 
        ref_lang = lang_pair[3:]

        if not os.path.isdir(os.path.join(output_folder, 'system-outputs', lang_pair)):
            os.mkdir(os.path.join(output_folder, 'system-outputs', lang_pair))

        lang_subfolder = os.path.join(hyp_folder, lang_pair)
        for system_file in os.listdir(lang_subfolder): # something like: newstest2014.cu-moses.3383.cs-en
            system_name = '.'.join(system_file.split('.')[1:3]) # cu-moses.3383
            input_file = os.path.join(lang_subfolder, system_file)
            output_file = os.path.join(output_folder, 'system-outputs', lang_pair, '%s.%s.%s' % (system_name, lang_pair, ref_lang))
            embed(input_file, ref_lang, output_file)




def cossim_write(file, ref_corpus, hyp_corpus):
    """ 
    Given a reference and system-output corpus, writes info to the given file in the following format.
    Format per line: <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SEGMENT NUMBER>   <SEGMENT SCORE> 
    For example: BEER    cs-en   newstest2014    cu-moses.3383   1       248.796234

    """
    metric_name = "SES"
    test_set = "newstest2018"

    hyp_info = os.path.basename(hyp_corpus).split('.')
    system_name = '.'.join(hyp_info[0:2])
    lang_pair = hyp_info[2]

    dim = 1024
    lang1 = np.fromfile(ref_corpus, dtype=np.float32, count=-1)
    lang2 = np.fromfile(hyp_corpus, dtype=np.float32, count=-1)
    lang1.resize(lang1.shape[0] // dim, dim)
    lang2.resize(lang2.shape[0] // dim, dim)

    for i in range(lang1.shape[0]):
        sim_str = str(1-dist.cosine(lang1[i], lang2[i]))
        file.write('\t'.join([metric_name, lang_pair, test_set, system_name, str(i+1), sim_str, 'non-emsemble', 'yes' + '\n']))


def cossim(ref_corpus, hyp_corpus):
    """ 
    Given two corpora, return cosine similarities for each example
    """
    dim = 1024
    lang1 = np.fromfile(ref_corpus, dtype=np.float32, count=-1)
    lang2 = np.fromfile(hyp_corpus, dtype=np.float32, count=-1)
    lang1.resize(lang1.shape[0] // dim, dim)
    lang2.resize(lang2.shape[0] // dim, dim)

    cossims = np.zeros(lang1.shape[0])
    print(lang1.shape, lang2.shape)

    for i in range(lang1.shape[0]):
        # print(i, lang1[i], lang2[i])
        cossims[i] = 1-dist.cosine(lang1[i], lang2[i])

    return cossims


def write_ses_score():
    # subm_folder = '%s/u2/metrics/wmt14-metrics-task/submissions/SES/' % os.environ['HOME']
    subm_folder = '%s/u2/metrics/wmt18-metrics-task-package/submissions-as-received/SES/' % os.environ['HOME']
    if not os.path.isdir(subm_folder):
        os.mkdir(subm_folder)
    f = open(os.path.join(subm_folder,'ses.seg.score'), 'w')
    

    enc_ref_folder = os.path.join(output_folder, 'references')

    for ref_file in os.listdir(enc_ref_folder):
        ref_path = os.path.join(enc_ref_folder, ref_file)

        lang_pair, _ = ref_file.split('.')

        enc_hyp_folder = os.path.join(output_folder, 'system-outputs', lang_pair)
        for hyp_file in os.listdir(enc_hyp_folder):
            hyp_path = os.path.join(enc_hyp_folder, hyp_file)
            cossim_write(f, ref_path, hyp_path)

    f.close()





def calc_ses(exp_name, exp_id, hyp_num):
    # if not os.isdir('../encodings/'):
    #     os.mkdir('../encodings/')
    dump_path = os.path.join('../dumped/', exp_name, exp_id)

    ref_file_exts = ['ref.en-fr.test.txt',
        'ref.en-fr.valid.txt',
        'ref.fr-en.test.txt',
        'ref.fr-en.valid.txt']

    hyp_file_exts = ['hyp%s.en-fr.test.txt' % str(hyp_num),
        'hyp%s.en-fr.valid.txt' % str(hyp_num),
        'hyp%s.fr-en.test.txt' % str(hyp_num),
        'hyp%s.fr-en.valid.txt' % str(hyp_num)]

    all_file_exts = ref_file_exts + hyp_file_exts

    for file_ext in all_file_exts:
        lang = file_ext.split('.')[1].split('-')[1]
        output_file = '.'.join(file_ext.split('.')[:-1])+'.enc'

        filepath = os.path.join(dump_path, file_ext)
        if not os.path.isfile(os.path.join(dump_path, output_file)):
            # print('notafile:%s', os.path.join(dump_path, output_file))
            embed(filepath, lang, os.path.join(dump_path, output_file))
            

    cossims = []
    for i in range(4):
        ref_enc = os.path.join(dump_path, '.'.join(ref_file_exts[i].split('.')[:-1])+'.enc')
        hyp_enc = os.path.join(dump_path, '.'.join(hyp_file_exts[i].split('.')[:-1])+'.enc')

        # cos = cossim(os.path.join(dump_path, ref_file_exts[i]), os.path.join(dump_path, hyp_file_exts[i]))
        cos = cossim(ref_enc, hyp_enc)
        with open(os.path.join(dump_path, ref_file_exts[i]), 'r') as f:
            ref_lines = f.readlines()
        with open(os.path.join(dump_path, hyp_file_exts[i]), 'r') as f:
            hyp_lines = f.readlines()

        cos_inds = np.argsort(cos)[::-1]
        # hyp_file = open(os.path.join(dump_path, hyp_file_exts[i]), 'r')
        with open(os.path.join(dump_path,'cos.' + '.'.join(ref_file_exts[i].split('.')[1:])), 'w') as f:
            for ind in cos_inds:
                f.write('\t'.join([str(cos[ind]), ref_lines[ind], hyp_lines[ind]]))
                

        lang_pair = ref_file_exts[i].split('.')[1]
        dataset = ref_file_exts[i].split('.')[2]
        print('SES %s %s: %.4f' % (lang_pair, dataset, np.mean(cos)))
        cossims.append(np.mean(cos))
    

    
parser = argparse.ArgumentParser(description='Language transfer')
parser.add_argument("--encode_refs", action="store_true",
                    help="Encode ref sentences (for WMT)")
parser.add_argument("--encode_hyps", action="store_true",
                    help="Encode hyp sentences (for WMT)")
parser.add_argument("--write_ses", action="store_true",
                    help="Write ses to file (for WMT)")
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")
parser.add_argument("--hyp_num", type=str, default="",
                    help="Hypothesis epoch num")
params = parser.parse_args()




if __name__ == "__main__":
    if params.encode_refs:
        encode_refs(os.listdir(ref_folder))
    elif params.encode_hyps:
        encode_hyps(os.listdir(hyp_folder))
    elif params.write_ses:
        write_ses_score()
    else:
        calc_ses(params.exp_name, params.exp_id, params.hyp_num)


# def tokenize(filepath, lang):
#     TOKENIZER = '~/u2/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl'
#     command = TOKENIZER + '-l %s < %s'
#     p = subprocess.Popen(command % (lang, filepath), stdout=subprocess.PIPE, shell=True)
#     result = p.communicate()[0].decode("utf-8")
#     print(result)
#     # if result.startswith('BLEU'):
#     #     return float(result[7:result.index(',')])
#     # else:
#     #     logger.warning('Impossible to parse BLEU score! "%s"' % result)
#     #     return -1





# if [ -z ${LASER+x} ] ; then
#   echo "Please set the environment variable 'LASER'"
#   exit
# fi

# if [ $# -ne 3 ] ; then
#   echo "usage embed.sh input-file language output-file"
#   exit
# fi

# ifile=$1
# lang=$2
# ofile=$3

# # encoder
# model_dir="${LASER}/models"
# encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
# bpe_codes="${model_dir}/93langs.fcodes"

# cat $ifile \
#   | python3 ${LASER}/source/embed.py \
#     --encoder ${encoder} \
#     --token-lang ${lang} \
#     --bpe-codes ${bpe_codes} \
#     --output ${ofile} \
#     --verbose
