# SDOH identification and extraction pipeline

## Summary

Code for study *Identification and extraction of social determinants of health: a robust solution*.

This study proposed a comprehensive pipeline in SDOH data classification and extraction. We provide evidence that the classification model outperforms benchmark models in SDOH category classification, and the extraction model can produce superior performance in extracting key sentences containing SDOH information.

## Usage

### Data proprocessing and purposeful sampling

- `data_cleaning.py`: break nursing notes to sentences, tokenize sentences

- `data_SDOH_expansion.py`: expand the SDOH dictionary

- `data_identifying.py` and `data_sampling.py`: generate SDOH corpus purposefully based on the SDOH dictionary 

- `data_join_pt.py`: add patiend id to the annotated data to split train/validation/test fold by patients

- `data_distribution.py`: explore data distributions

### Identification and Extraction

- `train.py`: train the BERT base classifier

- `evaluation.ipynb`: evaluate the BERT base classifier and compare with baseline models

- `extraction.ipynb`: extract key SDOH sentences based on classification result
