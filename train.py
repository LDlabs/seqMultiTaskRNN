"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

import task
from task import generate_trials, generate_datasetTensors, datasetGeneratorFromTaskDef, defineDatasetFormat
from network import Model, get_perf, Sequential_Model
import tools
from datetime import datetime as datetime
from tensorflow.python.ops import array_ops
import pdb
from seq_tools import compute_projection_matrices, compute_covariance


def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 2
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {  # factor to multiply delay periods during training
            'delay_fac': 1,
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 8192,  # changed from 512 jan 8th 2019
            # n_reps for testing
            'n_rep': 256,  # changed from 16 jan 8th 2019
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # orthogonalize separate task input, activity for sequential learning
            'sequential_orthog': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 1.,
            # Stopping cost
            'target_cost': 0,  # basically off
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1 + num_ring * n_eachring,
            # number of input units
            'n_input': 1 + num_ring * n_eachring + n_rule,
            # number of output units
            'n_output': n_eachring + 1,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # momentum for sgd_mom
            'momentum': 0.1,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
    }

    return hp


def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    # print(hp['rules'])
    for rule_test in hp['rules']:
        # rule_test = rule_train
        # for rule_test in hp['rule_trains']:

        n_rep = hp['n_rep']
        batch_size_test_rep = int(hp['batch_size_test'] / n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hp, 'random', batch_size=batch_size_test_rep, delay_fac=hp['delay_fac'])
            feed_dict = tools.gen_feed_dict(model, trial, hp)

            # import pdb
            # pdb.set_trace()

            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_' + rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_' + rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_' + rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # TODO: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_' + r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_' + r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    cost_tests_max = np.max([log['cost_' + r][-1] for r in rule_tmp])  # jan 4 2019
    log['cost_max'].append(cost_tests_max)  # jan 4 2019

    # Saving the model
    model.save()
    tools.save_log(log)

    return log


def do_eval_test(sess, model, rule):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp

    trial = generate_trials(rule, hp, 'test')
    feed_dict = tools.gen_feed_dict(model, trial, hp)
    c_lsq, c_reg, y_hat_test = sess.run(
        [model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)

    # Cost is first summed over time,
    # and averaged across batch and units
    # We did the averaging over time through c_mask
    perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
    sys.stdout.flush()

    return c_lsq, c_reg, perf_test


def display_rich_output(model, sess, step, log, model_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model, sess)
    rule_pair = ['contextdm1', 'contextdm2']
    save_name = '_atstep' + str(step)
    title = ('Step ' + str(step) +
             ' Perf. {:0.2f}'.format(log['perf_avg'][-1]))
    variance.plot_hist_varprop(model_dir, rule_pair,
                               figname_extra=save_name,
                               title=title)
    plt.close('all')


def train(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=True,
          load_dir=None,
          trainables=None,
          fixReadoutandBias=False,
          fixBias=False,
          ):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.

        rule_prob = np.array(
            [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))

    tools.save_hp(hp, model_dir)

    # Build the model
    with tf.device('gpu:0'):
        model = Model(model_dir, hp=hp)

        # Display hp
        for key, val in hp.items():
            print('{:20s} = '.format(key) + str(val))

        if fixReadoutandBias is True:
            my_var_list = [var for var in model.var_list if 'rnn/leaky_rnn_cell/kernel:0' in var.name]
            print(my_var_list)
        elif fixBias is True:
            my_var_list = [var for var in model.var_list if 'rnn/leaky_rnn_cell/kernel:0' in var.name or 'output/weights:0' in var.name]
        else:
            my_var_list = model.var_list

        model.set_optimizer(var_list=my_var_list)

        # Store results
        log = defaultdict(list)
        log['model_dir'] = model_dir

        # Record time
        t_start = time.time()

        # Use customized session that launches the graph as well
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # penalty on deviation from initial weight
            if hp['l2_weight_init'] > 0:
                anchor_ws = sess.run(model.weight_list)
                for w, w_val in zip(model.weight_list, anchor_ws):
                    model.cost_reg += (hp['l2_weight_init'] *
                                       tf.nn.l2_loss(w - w_val))

                model.set_optimizer(var_list=my_var_list)

            # partial weight training
            if ('p_weight_train' in hp and
                (hp['p_weight_train'] is not None) and
                    hp['p_weight_train'] < 1.0):
                for w in model.weight_list:
                    w_val = sess.run(w)
                    w_size = sess.run(tf.size(w))
                    w_mask_tmp = np.linspace(0, 1, w_size)
                    hp['rng'].shuffle(w_mask_tmp)
                    ind_fix = w_mask_tmp > hp['p_weight_train']
                    w_mask = np.zeros(w_size, dtype=np.float32)
                    w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                    w_mask = tf.constant(w_mask)
                    w_mask = tf.reshape(w_mask, w.shape)
                    model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
                model.set_optimizer(var_list=my_var_list)

            step = 0
            run_ave_time = []
            while step * hp['batch_size_train'] <= max_steps:
                try:
                    # Validation
                    if step % display_step == 0:
                        grad_norm = tf.global_norm(model.clipped_gs)
                        grad_norm_np = sess.run(grad_norm)
                        # import pdb
                        # pdb.set_trace()
                        log['grad_norm'].append(grad_norm_np.item())
                        log['trials'].append(step * hp['batch_size_train'])
                        log['times'].append(time.time() - t_start)
                        log = do_eval(sess, model, log, hp['rule_trains'])
                        # if log['perf_avg'][-1] > model.hp['target_perf']:
                        # check if minimum performance is above target
                        if log['perf_min'][-1] > model.hp['target_perf']:
                            print('Perf reached the target: {:0.2f}'.format(
                                hp['target_perf']))
                            break

                        if rich_output:
                            display_rich_output(model, sess, step, log, model_dir)

                    # Training

                    dtStart = datetime.now()
                    sess.run(model.train_step)
                    dtEnd = datetime.now()

                    if len(run_ave_time) is 0:
                        run_ave_time = np.expand_dims((dtEnd - dtStart).total_seconds(), axis=0)
                    else:
                        run_ave_time = np.concatenate((run_ave_time, np.expand_dims((dtEnd - dtStart).total_seconds(), axis=0)))

                    # print(np.mean(run_ave_time))
                    # print((dtEnd-dtStart).total_seconds())

                    step += 1

                    if step < 10:
                        model.save_ckpt(step)

                    if step < 1000:
                        if step % display_step / 10 == 0:
                            model.save_ckpt(step)

                    if step % display_step == 0:
                        model.save_ckpt(step)

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

            print("Optimization finished!")


def train_sequential_orthogonalized(
        model_dir,
        rule_trains,
        hp=None,
        max_steps=1e7,
        display_step=500,
        rich_output=False,
        ruleset='mante',
        applyProj='both',
        seed=0,
        nEpisodeBatches=100,
        projGrad=True,
        alpha=0.001,
        fixReadout=False):
    '''Train the network sequentially.

    Args:
        model_dir: str, training directory
        rule_trains: a list of list of tasks to train sequentially
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps for each list of tasks
        display_step: int, display steps
        ruleset: the set of rules to train
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    '''

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = rule_trains
    # Get all rules by flattening the list of lists
    # hp['rules'] = [r for rs in rule_trains for r in rs]
    hp['rules'] = rule_trains

    # save some other parameters
    hp['alpha_projection'] = alpha
    hp['max_steps'] = max_steps

    # Number of training iterations for each rule
    rule_train_iters = [max_steps for _ in rule_trains]

    tools.save_hp(hp, model_dir)
    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Build the model
    model = Sequential_Model(model_dir, projGrad=projGrad, applyProj=applyProj, hp=hp)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    def relu(x):
        return x * (x > 0.)

    # -------------------------------------------------------

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            raise NotImplementedError()

        # Looping
        step_total = 0
        taskNumber = 0

        if fixReadout is True:
            my_var_list = [var for var in model.var_list if 'rnn/leaky_rnn_cell/kernel:0' in var.name]
        else:
            my_var_list = [var for var in model.var_list if 'rnn/leaky_rnn_cell/kernel:0' in var.name or 'output/weights:0' in var.name]

        # initialise projection matrices
        input_proj = tf.zeros((hp['n_rnn'] + hp['n_input'], hp['n_rnn'] + hp['n_input']))
        activity_proj = tf.zeros((hp['n_rnn'], hp['n_rnn']))
        output_proj = tf.zeros((hp['n_output'], hp['n_output']))
        recurrent_proj = tf.zeros((hp['n_rnn'], hp['n_rnn']))

        for i_rule_train, rule_train in enumerate(hp['rule_trains']):

            step = 0

            model.set_optimizer(activity_proj=activity_proj, input_proj=input_proj, output_proj=output_proj, recurrent_proj=recurrent_proj, taskNumber=taskNumber, var_list=my_var_list, alpha=alpha)

            # Keep training until reach max iterations
            while (step * hp['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
                # Validation
                if step % display_step == 0:
                    trial = step_total * hp['batch_size_train']
                    log['trials'].append(trial)
                    log['times'].append(time.time() - t_start)
                    log['rule_now'].append(rule_train)
                    log = do_eval(sess, model, log, rule_train)
                    if log['perf_avg'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        break

                # Training
                # rule_train_now = hp['rng'].choice(rule_train)

                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                    rule_train, hp, 'random',
                    batch_size=hp['batch_size_train'], delay_fac=hp['delay_fac'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)

                # update model
                sess.run(model.train_step, feed_dict=feed_dict)

                # # Get the weight after train step
                # v_current = sess.run(model.var_list)

                step += 1
                step_total += 1

                if step % display_step == 0:
                    model.save_ckpt(step_total)

            # ---------- save model after its completed training the current task ----------
            model.save_after_task(taskNumber)

            # ---------- generate task activity for continual learning -------
            trial = generate_trials(
                rule_train, hp, 'random',
                batch_size=hp['batch_size_test'], delay_fac=hp['delay_fac'])

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            eval_h, eval_x, eval_y, Wrec, Win = sess.run([model.h, model.x, model.y, model.w_rec, model.w_in], feed_dict=feed_dict)
            full_state = np.concatenate([eval_x, eval_h], -1)

            # get weight matrix after current task
            Wfull = np.concatenate([Win, Wrec], 0)

            # joint covariance matrix of input and activity
            Shx_task = compute_covariance(np.reshape(full_state, (-1, hp['n_rnn'] + hp['n_input'])).T)

            # covariance matrix of output
            Sy_task = compute_covariance(np.reshape(eval_y, (-1, hp['n_output'])).T)

            # get block matrices from Shx_task
            # Sh_task = Shx_task[-hp['n_rnn']:, -hp['n_rnn']:]
            Sh_task = np.matmul(np.matmul(Wfull.T, Shx_task), Wfull)

            # ---------- update stored covariance matrices for continual learning -------
            if taskNumber == 0:
                input_cov = Shx_task
                activity_cov = Sh_task
                output_cov = Sy_task
            else:
                input_cov = taskNumber / (taskNumber + 1) * input_cov + Shx_task / (taskNumber + 1)
                activity_cov = taskNumber / (taskNumber + 1) * activity_cov + Sh_task / (taskNumber + 1)
                output_cov = taskNumber / (taskNumber + 1) * output_cov + Sy_task / (taskNumber + 1)

            # ---------- update projection matrices for continual learning ----------
            activity_proj, input_proj, output_proj, recurrent_proj = compute_projection_matrices(activity_cov, input_cov, output_cov, input_cov[-hp['n_rnn']:, -hp['n_rnn']:], alpha)

            # update task number
            taskNumber += 1

        print("Optimization Finished!")
