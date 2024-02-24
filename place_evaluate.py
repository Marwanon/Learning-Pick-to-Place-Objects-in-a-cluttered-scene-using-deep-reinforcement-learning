#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 04:51:01 2020

@author: marwan
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('--session_directory', dest='session_directory', action='store', type=str, help='path to session directory for which to measure performance')
parser.add_argument('--method', dest='method', action='store', type=str, help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
parser.add_argument('--num_obj_complete', dest='num_obj_complete', action='store', type=int, help='number of objects picked before considering task complete')

args = parser.parse_args()
session_directory = args.session_directory
method = args.method
num_obj_complete = args.num_obj_complete

# Parse data from session (action executed, reward values)
# NOTE: reward_value_log just stores some value which is indicative of successful grasping, which could be a class ID (reactive) or actual reward value (from MDP, reinforcement)
transitions_directory = os.path.join(session_directory, 'transitions')
executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
max_iteration = executed_action_log.shape[0]
executed_action_log = executed_action_log[0:max_iteration,:]
reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
#grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
reward_value_log = reward_value_log[0:max_iteration]
clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
place_success_log = np.loadtxt(os.path.join(transitions_directory, 'place-success.log.txt'), delimiter=' ')
grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
grasp_success_log = grasp_success_log[0:max_iteration]

# work around a bug where the clearance steps were written twice per clearance
place_success_log = np.unique(place_success_log)
max_trials_place = len(place_success_log)
place_success_log = np.concatenate((np.asarray([0]), place_success_log), axis=0).astype(int)

clearance_log = np.unique(clearance_log)
max_trials = len(clearance_log)
clearance_log = np.concatenate((np.asarray([0]), clearance_log), axis=0).astype(int)

num_actions_before_completion = clearance_log[1:(max_trials+1)] - clearance_log[0:(max_trials)]

#
place_success_rate = np.zeros((max_trials_place))
place_num_success = np.zeros((max_trials))
for trials_place in range(1,len(place_success_log)):
     tmp_executed_place_action_log = grasp_success_log[place_success_log[trials_place-1]:place_success_log[trials_place]]
     tmp_place_reward_value_log = reward_value_log[place_success_log[trials_place-1]:place_success_log[trials_place]]
     tmp_place_attempt_ind = np.argwhere(tmp_executed_place_action_log == 1)
     start_place = place_success_log[trials_place-1]
     end_place = place_success_log[trials_place]
     if method == 'reinforcement':
        tmp_num_place_success = np.sum(tmp_place_reward_value_log[tmp_place_attempt_ind] >= 0.5) # Reward value for successful grasping is anything larger than 0.5 (reinforcement policy)
        place_num_success[trials_place-1] = tmp_num_place_success
#        nt = end_place - start_place
        place_success_rate[trials_place-1] = float( place_num_success[trials_place-1])/float(max_trials_place) 
        

        
print('max_trials_place:' + str(max_trials_place))   
print('number of success iteration: %f x %f' % (float(tmp_num_place_success), float(max_trials_place)))   
print('Average %% place success per clearance: %2.1f' % (np.mean(place_success_rate[trials_place-1])*100))
print('Average %% action efficiency: %2.1f' % (100*np.mean((place_success_rate)))) 

