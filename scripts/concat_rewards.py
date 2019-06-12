import numpy as np
import pandas as pd
from glob import glob
import json
import sys, os

path_results = sys.argv[1]
files_path = os.path.abspath(path_results)

path = str(files_path) + '/'+'action_id_*.json'
df = pd.DataFrame()
# Concatenate rewards
for json_f in glob(path):
    with open(json_f) as f:
        config_data = json.load(f)
        config_df = pd.DataFrame(config_data['results']['rewards'])
        config_df.columns = ['rewards_'+str(config_data['action_index'])]
        df = pd.concat([df ,config_df], axis=1)

# Insert datetime and scenarios
for json_f in glob(path):
    with open(json_f) as f:
        df.insert(0, 'datetimes', config_data['results']['datetimes'])
        df.insert(1, 'scenario', config_data['results']['scenario'])
    break

df.to_csv(os.path.join(files_path,'concat_rewards.csv'))
