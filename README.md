# Variational Unsupervised Machine Translation

This is my master thesis project, and concerns neural machine translation using only monolingual corpora, a.k.a. Unsupervised NMT.

## Background
The main idea of unsupervised NMT is to translate from one language to another without any data with sentence-to-sentence translations. In other words, the only data given is text from both languages.  
This has been previously done with surprising success, most recently by [facebookresearch/UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT), achieving BLEU scores of 24+ on EN-FR and FR-EN.
This model builds on the work from UnsupervisedMT by adding variational encoding. This is added for the ability to generate multiple translations, and has the added benefit of being able to paraphrase.

I presented a poster at [CLIN 29](https://www.let.rug.nl/clin29/), which can be seen in `PosterCLIN.pdf`. Since then, the BLEU results have been improved with word duplicate noise, matching the performance of the non-variational model.


## How to run

### Preprocessing
Follow the same preprocessing steps as in [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT).
For the paraphrase dataset, 

Use the same arguments as specified in UnsupervisedMT, under the section "Train the NMT model" to run a normal, non-variational model. For variational encoding, these arguments have been added:

```
## Variational params
--variational True                          # Uses the varational model
--lambda_vae 1                              # Variational loss coefficient
--vae_samples 5                             # How many samples to take during back-translation

## Extra noise params
--word_duplicate 0.1                        # duplicate random words (helps VAE training)
```

Note: the "vae_samples" argument is typically best as a mulitple of the number of on-the-fly processors you are using. So far, I have only been able to test this on 10 processors (compared to the 30 used in UnsupervisedMT), and 5 samples works the best for this. 


## Sentence Embedding Similarity (SES)
To get SES scores on a text or reproduce our correlation results on WMT18, you can run ses_score.py. We assume you have [LASER](https://github.com/facebookresearch/LASER) installed to tools/LASER/

### Files from anywhere
If you simply want to run SES for a reference and hypothesis file:
```
python src/ses_score.py --ref ref_filepath --hyp hyp_filepath --ref_lang ref_language --hyp_lang hyp_language
```
The flags --ref_lang and --hyp_lang are only necessary if you care about which language MOSES uses for tokenization.

If you want to save the LASER encodings for later (faster) reevaluation:
```
python src/ses_score.py --ref ref_filepath --hyp hyp_filepath --save_ref --save_hyp
```
The LASER encodings will be saved to the same folder(s) as the reference or hypothesis files. 


### Files from an experiment
If you want to get SES scores for any epoch in an experiment, run:
```
python src/ses_score.py --exp_name name_of_the_experiment --exp_id experiment_id --hyp_num epoch_number
```
For all epochs, set --hyp_num to "all"


### WMT18 files
To reproduce WMT18 correlation scores, we assume you have downloaded the submitted data and metrics task package to the respective folders:
metrics/wmt18-submitted-data/
metrics/wmt18-metrics-task-package/

The data and package can be found at: http://www.statmt.org/wmt18/results.html

For WMT18 sentence-level scores:
```
python src/ses_score.py --encode_refs --encode_hyps --write_ses_score
```
For WMT18 system-level scores:
```
python src/ses_score.py --encode_refs --encode_hyps --write_ses_score --sys_score
```


