import os
os.environ["LASER"]="%s/u2/tools/LASER/" % os.environ['HOME']
import subprocess
import numpy as np
import scipy.spatial.distance as dist

# import source.embed as embed

ref_folder = '%s/u2/metrics/wmt14-data/txt/references/' % os.environ['HOME']
hyp_folder = '%s/u2/metrics/wmt14-data/txt/system-outputs/newstest2014/' % os.environ['HOME']
output_folder = '%s/u2/metrics/encodings/' % os.environ['HOME']


def embed(input_file, lang, output_file):
    ENCODER = os.environ['LASER']+'models/bilstm.93langs.2018-12-26.pt'
    BPE_CODES = os.environ['LASER']+'models/93langs.fcodes'
    EMBEDDER = 'python3 '+os.environ['LASER']+'source/embed.py '
    command = EMBEDDER + '< %s --encoder %s --token-lang %s --bpe-codes %s --output %s --verbose'
    p = subprocess.call(command % (input_file, ENCODER, lang, BPE_CODES, output_file), shell=True)
    # result = p.communicate()[0].decode("utf-8")
    # print(result)

def encode_refs():
    print('Encoding reference files...')
    # parse and encode references
    for filename in os.listdir(ref_folder): # something like: newstest2014-csen-ref.cs
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


def encode_hyps():
    print('Encoding hypothesis files...')
    # parse and encode system-outputs
    for lang_pair in os.listdir(hyp_folder): 
        ref_lang = lang_pair[3:]

        if not os.path.isdir(os.path.join(output_folder, 'system-outputs', lang_pair)):
            os.mkdir(os.path.join(output_folder, 'system-outputs', lang_pair))

        lang_subfolder = os.path.join(hyp_folder, lang_pair)
        for system_file in os.listdir(lang_subfolder): # something like: newstest2014.cu-moses.3383.cs-en
            system_name = '.'.join(system_file.split('.')[1:3]) # cu-moses.3383
            input_file = os.path.join(lang_subfolder, system_file)
            output_file = os.path.join(output_folder, 'system-outputs', lang_pair, '%s.%s.%s' % (system_name, lang_pair, ref_lang))
            embed(input_file, ref_lang, output_file)


encode_refs()
encode_hyps()


def cossim(file, corpus1, corpus2):

    dim = 1024
    lang1 = np.fromfile(corpus1, dtype=np.float32, count=-1)
    lang2 = np.fromfile(corpus2, dtype=np.float32, count=-1)
    lang1.resize(lang1.shape[0] // dim, dim)
    lang2.resize(lang2.shape[0] // dim, dim)

    for i in range(lang1.shape[0]):
        file.write(str(1-dist.cosine(lang1[i], lang2[i]))+'\n')


def write_ses_score():
    subm_folder = '%s/u2/metrics/wmt14-metrics-task/submissions/SES/' % os.environ['HOME']
    if not os.path.isdir(subm_folder):
        os.mkdir(subm_folder)
    
    f = open(os.path.join(subm_folder,'ses.seg.score'), 'w')
    # for 


    f.close()



def tokenize(filepath, lang):
    TOKENIZER = '~/u2/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl'
    command = TOKENIZER + '-l %s < %s'
    p = subprocess.Popen(command % (lang, filepath), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    print(result)
    # if result.startswith('BLEU'):
    #     return float(result[7:result.index(',')])
    # else:
    #     logger.warning('Impossible to parse BLEU score! "%s"' % result)
    #     return -1




# corpus1 = sys.argv[1]
# corpus2 = sys.argv[2]
# output = sys.argv[3]

# dim = 1024
# lang1 = np.fromfile(corpus1, dtype=np.float32, count=-1)
# lang2 = np.fromfile(corpus2, dtype=np.float32, count=-1)
# lang1.resize(lang1.shape[0] // dim, dim)
# lang2.resize(lang2.shape[0] // dim, dim)

# with open(output, 'w') as f:
#         for i in range(lang1.shape[0]):
#                 f.write(str(1-dist.cosine(lang1[i], lang2[i]))+'\n')







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
