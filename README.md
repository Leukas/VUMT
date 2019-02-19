# UNMT (Unsupervised Neural Machine Translation)

This is my ongoing master thesis project, and concerns neural machine translation using only monolingual corpora. 

## Background
The main idea of UNMT is to translate from one language to another without any data with sentence-to-sentence translations. In other words, the only data given is text from both languages.  
This has been previously done with surprising success, most recently by [facebookresearch/UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT), achieving BLEU scores of 24+ on EN-FR and FR-EN.
This model builds on the work from UnsupervisedMT by adding variational encoding. This is added for the ability to generate multiple translations, and has the added benefit of being able to paraphrase.


## How to run
Use the same arguments as specified in UnsupervisedMT, under the section "Train the NMT model" to run a normal, non-variational model. For variational encoding, these arguments have been added:

```
## Variational params
--variational True                          # Uses the varational model
--lambda_vae 1                              # Variational loss coefficient
--vae_samples 5                             # How many samples to take during back-translation
--eval_vae False                            # Evaluate the variatonal model 

## Extra noise params
--word_duplicate 0.1                        # duplicate random words (helps VAE training)
```

Note: the "vae_samples" argument is typically best as a mulitple of the number of on-the-fly processors you are using. So far, I have only been able to test this on 10 processors (compared to the 30 used in UnsupervisedMT), and 5 samples works the best for this. 

To evaluate the variational component, set "eval_vae" to True. This will evaluation without training, and currently samples at increasing distances from the mean, for both translation and paraphrasing. Currently, this only creates the output translation and paraphrasing files, but this will be updated soon to choose the best samples and evaluate BLEU automatically. 

## 