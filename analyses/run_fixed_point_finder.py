from __future__ import absolute_import, division, print_function
import os
import pdb
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys
import getpass

#################################################################
# Setup path to data and code
##################################################################

PATH_NET = os.path.dirname(os.getcwd())
sys.path.insert(0, PATH_NET)

# directory where example network is stored
model_dir_all = os.path.join(PATH_NET, 'data', 'trained_models', '0')

from task import generate_trials
from network import FixedPoint_Model
import tools

from RecurrentWhisperer import RecurrentWhisperer

##################################################################
# set up path to fixed point finder (EDIT THIS)
##################################################################

PATH_TO_FIXED_POINT_FINDER = '/path/to/your/directory/fixed-point-finder/'

# add fixed point finder to path
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

##################################################################
# run fixed point finder
##################################################################


NOISE_SCALE = 0.05  # 0.5 # Standard deviation of noise added to initial states
N_INITS = 1000  # The number of initial states to provide


task_list = ['delaygo', 'delayanti', 'fdgo', 'fdanti']

##################################################################


def add_unique_to_inputs_list(dict_list, key, value):
    for d in range(len(dict_list)):
        if (dict_list.values()[d] == value).all():
            return False, dict_list

    dict_list.update({key: value})
    return True, dict_list


def get_filename(trial, epoch, t):
    ind_stim_loc = 180 * trial.y_loc[-1, t] / np.pi
    filename = trial.epochs.keys()[epoch] + '_' + str(round(ind_stim_loc, 2))

    return filename, ind_stim_loc


for rule in task_list:
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:
        model.restore()
        model._sigma = 0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial
        trial = generate_trials(rule, hparams, mode='test', noise_on=False, batch_size=40)  # get feed_dict
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict)  # (n_time, n_condition, n_neuron)

        ##################################################################
        # get shapes
        n_steps, n_trials, n_input_dim = np.shape(trial.x)
        n_rnn = np.shape(h_tf)[2]
        n_output = np.shape(y_hat_tf)[2]

        # Fixed point finder hyperparameters
        # See FixedPointFinder.py for detailed descriptions of available
        # hyperparameters.
        fpf_hps = {}
        alr_dict = ({'decrease_factor': .95, 'initial_rate': 1})

        n_epochs = len(trial.epochs)
        for epoch in range(n_epochs):
            e_start = max([0, trial.epochs.values()[epoch][0]])
            end_set = [n_steps, trial.epochs.values()[epoch][1]]
            e_end = min(x for x in end_set if x is not None)

            n_inputs = 0
            input_set = {str(n_inputs): np.zeros((1, n_input_dim))}

            for t in range(0, np.shape(h_tf)[1], 10):  # Set which trials you want to find fixed points on

                inputs = np.squeeze(trial.x[e_start, t, :])
                inputs = inputs[np.newaxis, :]
                inputs_big = inputs[np.newaxis, :]

                unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)

                if unique_input:
                    n_inputs += 1
                    input_set[str(n_inputs)] = inputs

                    fpf = []
                    fpf = FixedPointFinder(model.cell, sess, alr_hps=alr_dict, method='joint', verbose=True, **fpf_hps)  # do_compute_input_jacobians = True , q_tol = 1e-1, do_q_tol = True

                    example_predictions = {'state': np.transpose(h_tf, (1, 0, 2)),  # [0:90,0:1,:]
                                           'output': np.transpose(y_hat_tf, (1, 0, 2))}

                    initial_states = fpf.sample_states(example_predictions['state'][:, :, :],  # specify T inds removed e_start:e_end
                                                       n_inits=N_INITS,
                                                       noise_scale=NOISE_SCALE)
                    # Run the fixed point finder
                    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

                    if unique_fps.xstar.shape[0] > 0:

                        all_fps = {}
                        all_fps = {'xstar': unique_fps.xstar,
                                   # 'J_inputs':unique_fps.J_inputs,
                                   'J_xstar': unique_fps.J_xstar,
                                   'qstar': unique_fps.qstar,
                                   'inputs': unique_fps.inputs,
                                   'epoch_inds': range(e_start, e_end),
                                   'noise_var': NOISE_SCALE,
                                   'state_traj': example_predictions['state'],
                                   'out_dir': 180 * trial.y_loc[-1, t] / np.pi}

                        save_dir = os.path.join(model_dir_all, 'tf_fixed_pts_all_init', rule)
                        filename, ind_stim_loc = get_filename(trial, epoch, t)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        np.savez(os.path.join(save_dir, filename + '.npz'), **all_fps)
