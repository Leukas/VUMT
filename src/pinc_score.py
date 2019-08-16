import argparse
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


parser = argparse.ArgumentParser(description='PINC score')
parser.add_argument("--src", type=str, default="",
                    help="Name of source file")
parser.add_argument("--hyp", type=str, default="",
                    help="Name of hypothesis file")
parser.add_argument("--ref", type=str, default="",
                    help="Name of reference files (for BLEU)")
params = parser.parse_args()


def eval_nltk_bleu(ref, hyp):
    """
    Given texts of structure: [ref1, ref2, ref3], [hyp1, hyp2, hyp3]
    Convert to proper structure for corpus_bleu, and run it.
    """
    ref_bleu = [[r.split()] for r in ref]
    hyp_bleu = [h.split() for h in hyp]
    return corpus_bleu(ref_bleu, hyp_bleu)

def eval_pinc(src, hyp):
    pinc_sum = 0.0
    for i in range(len(src)):
        pinc_sum += pinc_score(src[i].split(), hyp[i].split())
    return pinc_sum / len(src)

def pinc_score(src_sen, hyp_sen, max_ngram=4):
    """
    PINC score, as defined in 
    "Collecting Highly Parallel Data for Paraphrase Evaluation" (Chen et al. 2011)
    """
    src_ngrams = {}
    for n in range(max_ngram):
        src_ngrams[n] = [src_sen[i:i+n+1] for i in range(len(src_sen)-n)]

    ngram_counts = np.zeros(max_ngram)
    ngram_totals = np.array([max(len(hyp_sen) - i + 1, 1) for i in range(1, max_ngram+1)])

    for n in range(max_ngram):
        for i in range(ngram_totals[n]):
            ngram_counts[n] += hyp_sen[i:n+i+1] in src_ngrams[n]

    pinc = np.sum(1 - ngram_counts/ngram_totals) / max_ngram
    return pinc

def read_file(filename):
    corpus = []
    with open(filename, 'r') as f:
        for line in f:
            corpus.append(line)

    return corpus

if __name__ == "__main__":

    src_corpus = read_file(params.src)
    hyp_corpus = read_file(params.hyp)
    print(eval_pinc(src_corpus, hyp_corpus))
    # print(eval_nltk_bleu())
