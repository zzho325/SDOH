from pathlib import Path
import os, sys
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import multiprocessing
from multiprocessing import Pool
import re

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

def parallelize_split(func, num_of_processes=8):
  pool = Pool(num_of_processes)
  data = pool.map(func, range(main_data_num // 100000 + 1))
  pool.close()
  pool.join()
  return data
  
def split(index):
  def clean(x):
    x = x.lower()
    x = re.sub(r'[^\w\s]', '', x)
    x = re.sub(r'\d+', '', x)
    x = re.sub(r'\b\w\b', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x.split()
  
  start = 100000 * index
  end = min((100000 * index + 99999), main_data_num)
  print(f'splitting {start} to {end}')
  row_groups = main_data[start:end+1].apply(lambda row: [[sentence, row['NUM'], row['source_id']] for sentence in row['RESULT_TVAL'].split('.') if sentence != ''], axis=1)
  res = [row for rows in row_groups for row in rows]
  for row in res:
    row[0] = ' '.join([w for w in clean(row[0]) if w not in stopwords])
    row[0] if row[0] != '' else np.nan
    row.append(nltk.word_tokenize(row[0]))
  df = pd.DataFrame(res)
  df.rename(columns={0: 'RPT_TEXT', 1: 'NUM', 2: 'source_id', 3: 'tokenized_sents'}, inplace=True)
  df.drop(df.columns[[0]], axis=1, inplace=True)
  df.dropna(inplace=True)
  df.to_pickle(BASE_DIR / f'sentence/sentence_{index}.pkl')

if __name__ == '__main__':
  # load data, convert data to df with one row for each sentence 
  files = os.listdir(BASE_DIR / 'data_pt')
  print(f'processing file {files[0]}')
  main_data = pd.read_csv(BASE_DIR / f'data_pt/{files[0]}', lineterminator='\n', header=None, names=['NUM', 'RESULT_TVAL', 'source_id'])
  for file in files[1:]:
    print(f'processing file {file}')
    main_data = pd.concat([main_data, pd.read_csv(BASE_DIR / f'data_pt/{file}', lineterminator='\n', header=None, names=['NUM', 'RESULT_TVAL', 'source_id'])], axis=0, sort=False)
  
  main_data_num = len(main_data)
  print(f'data loaded : {main_data_num} documents')
  stopwords = set(stopwords.words('english'))
  
  # break notes, clean, tokenize
  parallelize_split(split, num_of_processes=multiprocessing.cpu_count())
  print('cleaned sentence generated')
