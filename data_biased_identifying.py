from pathlib import Path
import os, sys
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import pickle
import json
import time

BASE_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

def parallelize_identify(func, files, num_of_processes=8):
  pool = Pool(num_of_processes)
  data = pd.concat(pool.map(func, files))
  pool.close()
  pool.join()
  return data

def identify(file):
  def contains_lexcion(row):
    for w in bigram[row['tokenized_sents']] + row['tokenized_sents']:
      if w in blacklist:
        return False
      if w in SDOH_dict:
        return True
    return False
  sentence = pd.read_pickle(BASE_DIR / f'sentence/{file}')
  print(f'processing {file}')
  return sentence[sentence.apply(contains_lexcion, axis=1)]

if __name__ == '__main__':
  files = os.listdir(BASE_DIR / 'sentence')
  start_time = time.time()
  SDOH_dict = set()
  SDOH_dict.update(
    [
      # Income and other
      'veteran',
      'unemployed',
      'selfemployed',
      'losing_job',
      'lost_job',
      'jail',
      'prison',
      'criminal',
      'trespassing',
      'released_prison',
      'probation',
      'welfare',
      'finances',
      'financial_concerns',
      'financial_reasons',
      'concern_decompensation',
      'financially',
      'financial_strain',
      'payments',
      'cost',
      'lack_resources',
      # Housing
      'streets',
      'transitional_housing',
      'flooded',
      'homelessness',
      'homeless',
      'lived_homeless',
      'homeless_shelter',
      'friends_house',
      'infested',
      'black_mold',
      'landlord',
      'motel',
      'pay_rent',
      'mothers_house',
      'shelter',
      # Insurance
      'insurance_reasons',
      'expensive',
      'medicaid',
      'cheaper',
      'copay',
      'insurance_issues',
      'lost_insurance',
      'lack_insurance',
      'uninsured',
      # Abuse
      'abuse',
      'hard_liquor',
      'addiction',
      'alcohol',
      'alcoholic_beverage',
      'alcoholism',
      'beers',
      'couple_beers',
      'beer',
      'binge_drinking',
      'blackout',
      'binge',
      'drinking_heavily',
      'drinking_beer',
      'alcoholic_drink',
      'drinker',
      'beer_pint',
      'dependence',
      'gin',
      'hangover',
      'alcholic_drink',
      'drank_vodka',
      'alcohol_intoxication',
      'bottle_wine',
      'intoxication',
      'glass_wine',
      'drank_wine',
      'liquor',
      'margarita',
      'bourbon',
      'rum',
      'scotch',
      'sober',
      'pack_beer',
      'drank_beers',
      'tequila',
      'vodka',
      'whiskey',
      'wine',
      'weed',
      'pot',
      'ecstasy',
      'cocaine',
      'recreational_drug',
      'cocaine_heroin',
      'abusecocaine',
      'cocain',
      'marijuana_cocaine',
      'crack_cocaine',
      'polysubstance_abuse',
      'drug_abuse',
      'drug',
      'heroin',
      'abusing',
      'smoked_crack',
      'smoker',
      'crack',
      'craving',
      'marijuana',
      'intentional_overdose',
      'drug_overdose',
      'narcotic',
      'opioid',
      'overdose',
      'opioid_abuse',
      'abuse_tobacco',
      'cigar',
      'cigarette',
      'still_smokes',
      'cig',
      'smoking_cigarettes',
      'abstain_tobacco',
      'outside_smoke',
      'smokes_cigarettes',
      'nicotine_abuse',
      'hookah',
      'nicotine',
      'admits_smoking',
      'smoke',
      'tobacco',
      'vape',
      'transgender',
      'female_transgender',
      'lesbian',
      'pronouns',
      # disability
      'disabled',
      'disability',
    ]
  )
  
  blacklist = [
    'laboratory'
  ]
  
  # load model
  with open(BASE_DIR / 'model/bigram.pkl','rb') as f:
    bigram = pickle.load(f)
  print('bigram loaded')

  # Identify Biased Sample
  bias_data = parallelize_identify(identify, files, num_of_processes=multiprocessing.cpu_count())
  bias_data.to_csv(BASE_DIR / 'bias_sample_id.csv', columns=['NUM', 'source_id'], index=False)
  end_time = time.time()
  print(f'takes {end_time - start_time} seconds')