from pathlib import Path
import os, sys
import pandas as pd

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

if __name__ == '__main__':
  files = os.listdir(BASE_DIR / 'data')
  print(f'processing file {files[0]}')
  main_data = pd.read_csv(BASE_DIR / f'data/{files[0]}', lineterminator='\n', header=None, names=['NUM', 'RESULT_TVAL', 'source_id'])
  for file in files[1:]:
    print(f'processing file {file}')
    main_data = pd.concat([main_data, pd.read_csv(BASE_DIR / f'data/{file}', lineterminator='\n', header=None, names=['NUM', 'RESULT_TVAL', 'source_id'])], axis=0, sort=False)

  bias_data = pd.read_csv(BASE_DIR / 'bias_sample_id.csv')
  print('bias_data loaded')
  
  # Exclude current corpus
  files = os.listdir(BASE_DIR / 'Annotation')
  print(f'processing file {files[0]}')
  sdh_notes = pd.read_excel(BASE_DIR / f'Annotation/{files[0]}', engine='openpyxl')
  for file in files[1:]:
    print(f'processing file {file}')
    sdh_notes = pd.concat([sdh_notes, pd.read_excel(BASE_DIR / f'Annotation/{file}', engine='openpyxl')], axis=0, sort=False)

  bias_data = pd.merge(
      bias_data,
      sdh_notes,
      how="left",
      on=["source_id", "NUM"],
      suffixes=("_x", "_y"),
      copy=True,
      indicator=True,
      validate=None,
  )

  bias_data = bias_data[bias_data['_merge'] == 'left_only'][['NUM', 'source_id']]

  print('current corpus excluded')
  
  bias_sample = pd.merge(
      bias_data,
      main_data,
      how="inner",
      on=["source_id", "NUM"],
      suffixes=("_x", "_y"),
      copy=True,
      indicator=False,
      validate=None,
  )

  bias_sample = bias_sample.drop_duplicates(subset=['RESULT_TVAL'])
  print(f'{len(bias_sample)} total selected')
  
  # Sampling 1000 notes
  if len(bias_sample) > 1000:
    bias_sample = bias_sample.sample(1000)
    print(f'{len(bias_sample)} total sampled')
  
  bias_sample.to_csv(BASE_DIR / 'bias_corpus.csv', index=False)
  print('corpus developed')