from __future__ import division

import os
import train

# set regularization parameters

# 1. L2 activity reg
l2_h_value = 1e-7

# 2. L2 weight reg
l2_w_value = 1e-5

# 3. value for alpha parameter in gradient projection for continual learning
alpha_value = 0.001

# set total number of iterations
max_iter_steps = 7.5e6

# save some info on what regularizers were used in folder structure
folder = 'example_run'
seed_val = 0

# which rules to train on
tasklabel = 'goantiset'
rule_set = ['fdgo', 'fdanti', 'delaygo', 'delayanti']

# set directory for saving model
homedir = 'data'
filedir = os.path.join(homedir, folder, tasklabel, 'proj_both', str(seed_val))

# perform model training
train.train_sequential_orthogonalized(filedir, projGrad=True, applyProj='both',
                                      alpha=alpha_value, seed=seed_val, max_steps=max_iter_steps, ruleset='all',
                                      rule_trains=rule_set,
                                      hp={'activation': 'relu',
                                          'l1_h': 0,
                                          'l2_h': l2_h_value,
                                          'l1_weight': 0,
                                          'l2_weight': l2_w_value,
                                          'l2_weight_init': 0,
                                          'n_eachring': 2,
                                          'n_output': 1 + 2,
                                          'n_input': 1 + 2 * 2 + 20,
                                          'delay_fac': 1,
                                          'sigma_rec': 0.05,
                                          'sigma_x': 0.1,
                                          'optimizer': 'sgd_mom',
                                          'momentum': 0.9,
                                          'learning_rate': 0.001,
                                          'use_separate_input': False},
                                      display_step=1000,
                                      rich_output=False)
