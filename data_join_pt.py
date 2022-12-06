from pathlib import Path
import os, sys
import pandas as pd

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

if __name__ == '__main__':

  files = os.listdir(BASE_DIR / 'data_pt')
  print(f'processing file {files[0]}')
  main_data = pd.read_csv(BASE_DIR / f'data_pt/{files[0]}', lineterminator='\n', header=None, names=['NUM', 'patient_id', 'RESULT_TVAL', 'source_id'])
  for file in files[1:]:
    print(f'processing file {file}')
    main_data = pd.concat([main_data, pd.read_csv(BASE_DIR / f'data_pt/{file}', lineterminator='\n', header=None, names=['NUM', 'patient_id', 'RESULT_TVAL', 'source_id'])], axis=0, sort=False)
  
  files = os.listdir(BASE_DIR / 'Annotation')
  for file in files:
    print(f'processing file {file}')
    sdh_notes = pd.read_excel(BASE_DIR / f'Annotation/{file}', engine='openpyxl')
    sdh_notes = pd.merge(
        sdh_notes,
        main_data,
        how="left",
        on=["source_id", "NUM"],
        suffixes=("_x", "_y"),
        copy=True,
        indicator=True,
        validate=None,
    )
    sdh_notes.drop(['_merge', 'RESULT_TVAL_y'], axis=1, inplace=True)
    sdh_notes.rename(columns={'RESULT_TVAL_x': 'RESULT_TVAL'}, inplace=True)
    sdh_notes.to_excel(BASE_DIR / f'Annotation_pt/{file}', index=False)