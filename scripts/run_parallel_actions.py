import numpy as np
import pandas as pd
import itertools
import copy
import os, re, sys
import json
import time
import csv
import argparse
import multiprocessing as mp
import pypownet.environment
import pypownet.runner
from datetime import datetime


class parallel_actions:
    def __init__(self):
        self.target_configurations = target_configurations
        self.input_path = None
        self.destination_path = None
        self.game_level = 'validation_backendonly'
        self.tot_iter = 5174
        self.save_files = True
        self.rel_dic = {}

    def set_environement(self):
        # Set pypownet environment
        return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(self.input_path),
                                           game_level=self.game_level,
                                           chronic_looping_mode='natural', start_id=0,
                                           game_over_mode='easy')

    def _get_config_names(self):
        # This method maps the target configuration dict
        # to impose a name for each one target with the
        # following format:
        # se_(sub_id)_(int)
        # sub_id     is the same sub id in pypownet
        # int        incremental int for each possible config in sub
        names_dic = {}
        for key in self.target_configurations:
            names = []
            names += [str(key) + '_' + str(i)  for i in range(len(self.target_configurations[key]))]
            names_dic[key] = names
        return names_dic

    def combinations(self):
        # Get all possible combination given the target dictionary
        names = self._get_config_names()
        names_combinations = list(itertools.product(*names.values()))
        print ('++ ++ ++ ++ ++ ++ ++ ++ ++ ++ +')
        print ('Total num of combinations: {}'.format(len(names_combinations)))
        # print ('Estimated time with 50 cores {:.4} hours \
        #        ~~considering 2 mins per action using validation set\n'.format((len(names_combinations)*2/(50*60))))
        return names_combinations

    def construct_final_action_dic(self):
        # This method computes the {i: {action_name: action}} for every
        # possible combination in target configuration where
        # i              is an index
        # action_name    is the name for a possible action (see get_config_names)
        # action         is a numpy array with the action

        env = self.set_environement()
        all_combinations = self.combinations()

        action_comb_dic = {}
        actions_table = {}
        i = 0
        for combination in all_combinations:
            indv_actions = []
            action = env.action_space.get_do_nothing_action(as_class_Action=True)
            for unitary in combination:
                # Decode se and tl names
                names = unitary.split('_')
                id, config_idx = int(names[1]), int(names[-1])
                # Unitary type name
                type = unitary[:2]
                if type == 'se':
                    action.set_substation_switches(substation_id=id,
                                                   new_values=self.target_configurations['se_' + str(id)][config_idx])
                    indv_actions.append(self.target_configurations['se_' + str(id)][config_idx])

                elif type == 'tl':
                    env.action_space.set_lines_status_switch_from_id(action=action,
                                                                     line_id=id,
                                                                     new_switch_value=self.target_configurations['tl_' + str(id)][config_idx][0])
                    indv_actions.append(self.target_configurations['tl_' + str(id)][config_idx][0])

            # Encode key name
            key_name = '_'.join(map(str,combination))
            # Create final nested dict {('0','encoded_keyname'): action_vector}}
            action_comb_dic[(i,key_name)] = copy.deepcopy(action)
            actions_table[i] = indv_actions
            i += 1
        return action_comb_dic, actions_table

    def sim_single_action(self, action):
        # This method return some observation variables
        # given only one action

        # Apply the action
        env = self.set_environement()
        env.game.apply_action(action)
        # Action do nothing
        action_do_nothing = env.action_space.get_do_nothing_action()
        rewards, observations, datetimes, sce = [], [], [], []
        for _ in range(self.tot_iter):
            obs, simulated_reward, *_ = env.step(action_do_nothing)
            tmp = env.get_current_datetime()
            tmp_str = tmp.strftime("%m/%d/%Y, %H:%M:%S")
            scenario = int(env.game.get_current_chronic_name())
            if obs is None:
                # print ('--> Obs None type <--')
                print ('--> Action will not be saved')
                self.save_files = False
                break

            # Save results in list
            rewards.append(simulated_reward); observations.append(list(obs))
            datetimes.append(tmp_str); sce.append(scenario)
        return datetimes, sce, rewards, observations

    def run_sim(self, key_name, action):
        dtimes, sces, rews, observs = self.sim_single_action(action)
        self.rel_dic['action_index'] = key_name[0]
        self.rel_dic['action_name'] = key_name[1]
        self.rel_dic['action_vec'] = action.as_array().tolist()
        self.rel_dic['datetimes'] = dtimes
        self.rel_dic['scenario'] = sces
        self.rel_dic['rewards'] = rews
        self.rel_dic['obs'] = observs
        f_name = 'action_id_' + str(key_name[0]) +'.json'
        full_path = os.path.join(self.destination_path, f_name)
        if self.save_files:
            with open(full_path, 'w') as f:
                json.dump(self.rel_dic, f)

        # Reset save file boolean
        self.save_files = True
        print ('++ Fin validation with action -> {}'.format(key_name[0]))

    def select_actions(self, start=0, end=2):
        # Select subset of actions given a range
        actions, actions_table = self.construct_final_action_dic()
        sample_actions = {k: v for j in range(start, end) for k, v in actions.items() if k[0]==j}

        # Save actions table as csv using pandas
        path = os.path.join(self.destination_path, 'actions_table.csv')
        self.actions_df = pd.DataFrame(actions_table.values(),
                                       index=actions_table.keys(),
                                       columns=self.target_configurations.keys())
        self.actions_df.to_csv(path)
        return sample_actions

    def run_parallel_sim(self, target_configurations,
                               start, end):

        assert end > start
        # Run iterate over actions
        print ('\nRunning the game...')
        print ('Number of processors available: {}\n'.format(mp.cpu_count()))

        # Init multiprocs pool
        pool = mp.Pool(mp.cpu_count())
        # Ini multiprocessing
        sample_actions = self.select_actions(start, end)

        start = time.time()
        results = pool.starmap(self.run_sim, [(key, action) for key, action in sample_actions.items()])
        # Close conextion
        pool.close()
        end = time.time()
        print ('Final time {:.5} m'.format((end-start)/60))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--game_path', metavar='GAME_PATH', type=str, required='True',
                        help='Path where the game level is located')

    parser.add_argument('-d', '--path_files', metavar='FILES_PATH', type=str, required='True',
                        help='Path where the json files should be saved them')

    parser.add_argument('-s', '--start', metavar='START_INDEX', type=int, default=0,
                        help='Min index to start computation')

    parser.add_argument('-e', '--end', metavar='END_INDEX', type=int, default=8639,
                        help='Max index to end computation')

    parser.add_argument('-a', '--actions_csv', metavar='ACTIONS_CSV', type=bool, default=False,
                        help='To save combination action csv')

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
                              'tl_4':[[0],[1]],
                              'tl_9':[[0],[1]],
                              'tl_17':[[0],[1]],
                              'tl_19':[[0],[1]],
                              }

    # Initialize the class
    sim = parallel_actions()
    sim.input_path = args.game_path
    sim.destination_path = args.path_files

    # Run parallel
    sim.run_parallel_sim(target_configurations, \
                         start=args.start, \
                         end=args.end)
