import numpy as np
import pandas as pd
import glob
import json
import sys, os

path_results = sys.argv[1]
files_path = os.path.join(path_results, action_id_*.json)

df = pd.DataFrame()
for json_f in glob(files_path):
    with open(json_f) as f:
        config_data = json.load(f)
        config_df = pd.DataFrame(config_data['results']['rewards'], index=config_data['results']['datetimes'])
        config_df.columns = ['rewards_'+str(config_data['action_index'])]
        df = pd.concat([df ,config_df], axis=1)

# df.head()
df.to_csv('concat_rewards.csv')


