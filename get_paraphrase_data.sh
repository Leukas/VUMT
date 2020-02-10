set -e

CODES=60000      # number of BPE codes
N_THREADS=24     # number of threads in data preprocessing

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

# create paths
mkdir -p $DATA_PATH/pp
mkdir -p $DATA_PATH/pp/coco

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
BPE_CODES=$MONO_PATH/bpe_codes
FULL_VOCAB=$MONO_PATH/vocab.en-fr.$CODES

COCO=$DATA_PATH/pp/coco
FILE_SET=$COCO/captions_val2014

# coco dataset
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P $COCO

# unzip dataset
unzip $COCO/annotations_trainval2014.zip

# split captions into src/refs
python load_caps.py $FILE_SET

# cut off words past 15th word (matching Gupta et al.)
awk '{ if (NF<16) { print } }' $FILE_SET.src.en > $FILE_SET.src-filtered.en
for i in {0..3}; do
  awk '{ if (NF<16) { print } }' $FILE_SET.ref$i.en > $FILE_SET.ref$i-filtered.en
done

# tokenize data, apply bpe, binarize
cat $FILE_SET.src-filtered.en | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $FILE_SET.src-filtered.en.tok
$FASTBPE applybpe $FILE_SET.src-filtered.en.tok.$CODES $FILE_SET.src-filtered.en.tok $BPE_CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $FILE_SET.src-filtered.en.tok.$CODES
for i in {0..3}; do
  cat $FILE_SET.ref$i-filtered.en | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $FILE_SET.ref$i-filtered.en.tok
  $FASTBPE applybpe $FILE_SET.ref$i-filtered.en.tok.$CODES $FILE_SET.ref$i-filtered.en.tok $BPE_CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $FILE_SET.ref$i-filtered.en.tok.$CODES
done

