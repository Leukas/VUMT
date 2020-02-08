# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$1
LANG=$2
SRC_TOK=$1.tok

BPE_CODES=$MONO_PATH/bpe_codes
SRC_VOCAB=$MONO_PATH/vocab.$LANG.$CODES
FULL_VOCAB=$MONO_PATH/vocab.en-fr.$CODES

# tokenize data
if ! [[ -f "$SRC_TOK" ]]; then
  echo "Tokenize data..."
  cat $SRC_RAW | $NORM_PUNC -l $LANG | $TOKENIZER -l $LANG -no-escape -threads $N_THREADS > $SRC_TOK
fi
echo "$LANG data tokenized in: $SRC_TOK"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES $SRC_VOCAB
fi
echo "BPE codes applied to EN in: $SRC_TOK.$CODES"


# # extract vocabulary
# if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
#   echo "Extracting vocabulary..."
#   $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
#   $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
#   $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
# fi
# echo "EN vocab in: $SRC_VOCAB"
# echo "FR vocab in: $TGT_VOCAB"
# echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
fi
echo "EN binarized data in: $SRC_TOK.$CODES.pth"

