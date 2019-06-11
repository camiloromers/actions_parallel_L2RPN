import numpy as np
import itertools
import os, re, sys
import argparse

parser = argparse.ArgumentParser(description='Get the index given a number of parallel machines')

# parser.add_argument('-p', '--game_path', metavar='AGENT_PATH', type=str, required='True',
#                     help='Path where the game level is located')

parser.add_argument('-n', '--n_machines', metavar='NUMBER_MACHINES', type=int, default=2, required='True',
                    help='Number of parallel machines to get the indexes')

args = parser.parse_args()


# Configurable by the user
'''
Convention for target_configuration:
se_[idx]    where idx depicts the id substations
tl_[idx]    where idx depicts the index corresponding
            to the lines_or_substations_ids
'''
target_configurations = {'se_6':[[0, 0, 0, 0, 0 ,0],
                                 [0, 0, 0, 0, 1, 1],
                                 [0, 1, 0, 0, 1, 1]],
                         'se_5':[[0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 1],
                                 [0, 0, 1, 0, 1]],
                         'se_4':[[0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 1, 0],
                                 [0, 0, 0, 1, 1, 0],
                                 [1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 1, 0]],
                         'se_9':[[0, 0, 0, 0, 0],
                                 [1, 0, 1, 0, 1],
                                 [0, 0, 1, 0, 1],
                                 [1, 1, 0, 1, 1]],
                         'se_2':[[0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 1, 0, 1],
                                 [1, 1, 0, 1, 0, 0]],
                          'tl_1':[[0],[1]],
                          'tl_2':[[0],[1]],
                          'tl_3':[[0],[1]],
                          'tl_4':[[0],[1]],
                          }


# Name of each target configuration
names_dic = {}
for key in target_configurations:
    names = []
    names += [str(key) + '_' + str(i)  for i in range(len(target_configurations[key]))]
    names_dic[key] = names

# All possible combinations
all_combs = list(itertools.product(*names_dic.values()))

# Create chunks of data
indexes = np.arange(len(all_combs))
chunks = np.array_split(indexes, args.n_machines)

# Print indexes
print ('++ ++ ++ ++ ++')
print ('Chunks indexes:')
for chunk in chunks:
    print ('Start: {:<4} - End {}'.format(chunk[0], chunk[-1]))
