from pathlib import Path
import os, sys
import pandas as pd
from gensim.models.phrases import Phrases
from gensim.models import Word2Vec
import multiprocessing
import pickle
import json
import time

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

if __name__ == '__main__':
  start_time = time.time()
  files = os.listdir(BASE_DIR / 'sentence')

  print(f'processing file {files[0]}')
  sentence = pd.read_pickle(BASE_DIR / f'sentence/{files[0]}')
  for file in files[1::2]:
    print(f'processing file {file}')
    sentence = pd.concat([sentence, pd.read_pickle(BASE_DIR / f'sentence/{file}')], axis=0, sort=False)

  sentence_num = len(sentence)
  print(f'sentence loaded : {sentence_num} documents')
  
  print(f'sentence sampled : {len(sentence)} documents')
  # Train Word2Vec
  EMB_DIM = 300
  
  bigram = Phrases(sentence.iloc[:,-1], min_count=5, threshold=10)
  with open(BASE_DIR / 'bigram.pkl','wb') as f:
      pickle.dump(bigram,f)
  print('bigram trained')
  
  w2v = Word2Vec(bigram[sentence.iloc[:,-1]], vector_size=EMB_DIM, window=5, min_count=5, negative=15, epochs=10, workers=multiprocessing.cpu_count())
  with open(BASE_DIR / 'w2v.pkl','wb') as f:
      pickle.dump(w2v,f)
  print('word2vec trained')

  # SDOH dictionary
  SDOH = {
    'income_and_other_financial_insecurity' : [
      'veteran',
      'unemployment',
      'losing_job',
      'lost_job',
      'jail',
      'welfare',
      'finances',
      'financial_concerns',
      'financially',
      'financial_strain',
      'prison',
      'probation',
      'criminal',
      'trespassing',
    ],
    'housing_insecurity' : [
      'streets',
      'transitional_housing',
      'flooded',
      'homeless',
      'infested',
      'landlord',
      'motel',
      'pay_rent',
      'shelter',
    ],
    'insurance_insecurity' : [
      'cheaper'
      'copay'
      'insurance_issues'
      'lost_insurance'
      'lack_insurance'
      'uninsured'
    ],
    'alcohol_use': [
      'abuse',
      'addiction',
      'alcohol',
      'alcoholic_beverage',
      'alcoholism',
      'beer',
      'binge_drinking',
      'blackout',
      'dependence',
      'gin',
      'hangover',
      'intoxication',
      'margarita',
      'rum',
      'scotch',
      'sober',
      'tequila',
      'vodka',
      'whiskey',
      'wine',
    ],
    'substance_use': [
      'weed',
      'pot',
      'ecstasy',
      'cocaine',
      'heroin',
      'crack',
      'craving',
      'marijuana',
      'intentional_overdose',
      'drug_overdose',
      'narcotic',
      'opioid',
      'overdose',
    ],
    'tobacco_use': [
      'cigar',
      'cigarette',
      'hookah',
      'nicotine',
      'smoke',
      'tobacco',
      'vape',
    ],
    'disability': [
      'disability'
    ],
  }
  SDOH_dict = set()
  SDOH_vocab = dict()
  word_vectors = w2v.wv

  for key in SDOH:
    for word in SDOH[key]:
      if word in w2v.wv.key_to_index:
        SDOH_dict.add(word)
        similar_list = word_vectors.similar_by_word(word)
        SDOH_dict.update([e[0] for e in similar_list[:10]])
        SDOH_vocab[word] = similar_list[:10]

  with open(BASE_DIR / 'SDOH_vocab.json', 'w') as f:
      f.write(json.dumps(SDOH_vocab))

  print('dictionary expanded')
  end_time = time.time()
  print(f'takes {end_time - start_time} seconds')