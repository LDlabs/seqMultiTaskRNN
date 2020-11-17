from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import json
from datetime import datetime as datetime
from tensorflow.python.ops import parallel_for as pfor
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from numpy import linalg as LA
import numpy.random as npr
from scipy import stats

import task
from task import generate_trials, rules_dict
from network import Model, get_perf, FixedPoint_Model 
import tools
import train

def gen_trials_from_model_dir(model_dir,rule,mode='test',noise_on = True,batch_size = 500):
    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
#         params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial
        trial = generate_trials(rule, hparams, mode=mode, noise_on=noise_on, batch_size =batch_size, delay_fac =1)
    return trial  

def gen_X_from_model_dir(model_dir,trial,d = []):
    model = Model(model_dir)
    with tf.Session() as sess:
        
        if len(d)==0:
            model.restore()
        else:
            model.saver.restore(sess,d)

        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        hparams = model.hp
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        x = np.transpose(h_tf,(2,1,0)) # h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return orthogonal complement of hidden unit activity to ouput projection matrix

def gen_X_from_model_dir_epoch(model_dir,trial,epoch,d = []):
    model = Model(model_dir)
    with tf.Session() as sess:
        
        if len(d)==0:
            model.restore()
        else:
            model.saver.restore(sess,d)

        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial       
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        
        if trial.epochs[epoch][1] is None:
            epoch_range = range(trial.epochs[epoch][0],np.shape(h_tf)[0])
        elif trial.epochs[epoch][0] is None:
            epoch_range = range(0,trial.epochs[epoch][1])
        else:
            epoch_range = range(trial.epochs[epoch][0],trial.epochs[epoch][1])

        x = np.transpose(h_tf[epoch_range,:,:],(2,1,0)) #h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return hidden unit activity

def restore_ckpt(model_dir, ckpt_n):
    ckpt_n_dir = os.path.join(model_dir,'ckpts/model.ckpt-' + str(int(ckpt_n)) + '.meta')
    model = Model(model_dir)
    with tf.Session() as sess:
        model.saver.restore(sess,ckpt_n_dir)
    return model

def find_ckpts(model_dir):
    s_all = []
    ckpt_n_dir = os.path.join(model_dir,'ckpts/')
    for file in os.listdir(ckpt_n_dir):
        if file.endswith('.meta'):
            m = re.search('model.ckpt(.+?).meta', file)
            if m:
                found = m.group(1)
            s_all = np.concatenate((s_all,np.expand_dims(abs(int(found)),axis=0)),axis = 0)
    return s_all.astype(int)

def name_best_ckpt(model_dir,rule):
    s_all = find_ckpts(model_dir)
    s_all_inds = np.sort(s_all)
    s_all_inds = s_all_inds.astype(int)
    fname = os.path.join(model_dir, 'log.json')
    
    with open(fname, 'r') as f:
        log_all = json.load(f)
        x = log_all['cost_'+rule]
           
    y = [x[int(j/1000)] for j in s_all_inds[:-1]]
    ind = int(s_all_inds[np.argmin(y)])
    return ind

def get_model_params(model_dir,ckpt_n_dir = []):

    model = Model(model_dir)
    with tf.Session() as sess:
        if len(ckpt_n_dir)==0:
            model.restore()
        else:
            model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]

    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]

    return w_in, b_in, w_out, b_out

def get_path_names():
    import getpass
    ui = getpass.getuser()
    if ui == 'laura':
        p = '/home/laura'
    elif ui == 'lauradriscoll':
        p = '/Users/lauradriscoll/Documents'
    return p

def take_names(epoch,rule,epoch_axes = [],h_epoch = []):
    epochs = ['stim1','delay1','go1']
    epoch_names = ['stimulus','memory','go']
    ei = [i for i,e in enumerate(epochs) if e==epoch]
    epoch_name = epoch_names[ei[0]]
    
    rules = ['fdgo','fdanti','delaygo','delayanti']
    rule_names = ['DelayPro','DelayAnti','MemoryPro','MemoryAnti']
    ri = [i for i,e in enumerate(rules) if e==rule]
    rule_name = rule_names[ri[0]]
    
    if len(epoch_axes)<1:
        epoch_axes_name = epoch_names[ei[0]]
    else:
        ei = [i for i,e in enumerate(epochs) if e==epoch_axes]
        epoch_axes_name = epoch_names[ei[0]]
    
    if len(h_epoch)==0:
        h_epoch = epoch
    
    return epoch_name, rule_name, epoch_axes_name, h_epoch

def plot_N(X, D, clist, linewidth = 1):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """
    cmap=plt.get_cmap('rainbow')
    S = np.shape(X)[0]
    
    for s in range(S):
        c = cmap(clist[s]/max(clist))
        X_trial = np.dot(X[s,:,:],D.T)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, linewidth = linewidth)
        plt.plot(X_trial[:,0],X_trial[:,1],'-',c = c, linewidth = linewidth)
        plt.plot(X_trial[0,0],X_trial[0,1],'.',c = c, linewidth = linewidth)

def plot_FP(X, D, eig_decomps, c='k'):
    """Plot activity is some 2D space.

        Args:
            X: Fixed points in #Fps x Neurons
            D: Neurons x 2 plotting dims
    
        """
    S = np.shape(X)[0]
    lf = 7
    rf = 7
    
    for s in range(S):
        
        X_trial = np.dot(X[s,:],D.T)
        
        n_arg = np.argwhere(eig_decomps[s]['evals']>1)+1
        if len(n_arg)>0:
            for arg in range(np.max(n_arg)):
                rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
                ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
                overlap = np.dot(rdots,ldots.T)
                r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
                plt.plot(r[0:4:2],r[1:4:2], c = c ,alpha = .2,linewidth = .5)
        
        n_arg = np.argwhere(eig_decomps[s]['evals']<.3)
        if len(n_arg)>0:
            for arg in range(np.min(n_arg),len(eig_decomps[s]['evals'])):
                rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
                ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
                overlap = np.dot(rdots,ldots.T)
                r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
                plt.plot(r[0:4:2],r[1:4:2],'b',alpha = .2,linewidth = .5)
            
        plt.plot(X_trial[0], X_trial[1], 'o', markerfacecolor = c, markeredgecolor = 'k', 
                 markersize = 6, alpha = .5)

def comp_eig_decomp(Ms, sort_by='real',
                                     do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.

  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.

  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = np.abs
  elif sort_by == 'real':
    sort_fun = np.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = LA.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = LA.pinv(R).T  # as columns      
      L = L[:, indices]
    decomps.append({'evals' : evals[indices], 'R' : R[:, indices],  'L' : L})
  
  return decomps

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rot_mat(theta):
    R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return R

def calc_R_angle(R):
    return np.arccos((np.trace(R)-1)/2)

def tranform_in_rPC(X,R,X_ss):
    Xr_ss = np.dot(R,X_ss.T).T
    Xr = np.dot(R,X.T).T
    if Xr_ss[1,1]>0:
        Xr = np.dot(Xr,np.array(((1,0),(0,-1))))
    return Xr

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def make_Jac_u_dot_delu(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
    n_tasks = len(task_set)
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                    J_np_u[r,trial_i,time_i,:,:] = sess.run(J_tf_u)
                    J_np_u_dot_delu[r,trial_i,time_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,time_i,:,:],delta_inputs.T))
                    
    return J_np_u_dot_delu

def make_Jac_x(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
    n_tasks = len(task_set)
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_x = np.zeros((n_tasks,len(trial_set),len(time_set),N,N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_x = pfor.batch_jacobian(F_context, states_tf, use_pfor=False)
                    J_np_x[r,trial_i,time_i,:,:] = sess.run(J_tf_x)
                    
    return J_np_x

def make_h_and_Jac(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):

    h_context_combined = []
    h_stim_early_combined = []
    h_stim_late_combined = []
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials('delaygo', hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        time_set = [trial.epochs['stim1'][0]] #beginning of stim period

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            # comparing Jacobians to proximity of hidden state across tasks
            # we focus on end of the context period, early, and late in the stim period
            h_context = np.reshape(h_tf[trial.epochs['stim1'][0]-1,trial_set,:],(1,-1)) # h @ end of context period
            h_stim_early = np.reshape(h_tf[trial.epochs['stim1'][0]+n_steps_early,trial_set,:],(1,-1)) # h @ 5 steps into stim
            h_stim_late = np.reshape(h_tf[trial.epochs['stim1'][1],trial_set,:],(1,-1)) # h @ end of stim period

            #concatenate activity states across tasks
            if h_context_combined == []:
                h_context_combined = h_context[np.newaxis,:]
                h_stim_late_combined = h_stim_late[np.newaxis,:]
                h_stim_early_combined = h_stim_early[np.newaxis,:]
            else:
                h_context_combined = np.concatenate((h_context_combined, h_context[np.newaxis,:]), axis=0)
                h_stim_late_combined = np.concatenate((h_stim_late_combined, h_stim_late[np.newaxis,:]), axis=0)
                h_stim_early_combined = np.concatenate((h_stim_early_combined, h_stim_early[np.newaxis,:]), axis=0)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                    J_np_u[r,trial_i,time_i,:,:] = sess.run(J_tf_u)
                    J_np_u_dot_delu[r,trial_i,time_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,time_i,:,:],delta_inputs.T))
                    
    return J_np_u_dot_delu, h_context_combined, h_stim_late_combined, h_stim_early_combined

def prep_procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.
    
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
  
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2
    
    return mtx1,mtx2

# def procrustes(mtx1, mtx2):
#     # transform mtx2 to minimize disparity
#     R, s = orthogonal_procrustes(mtx1, mtx2)
#     mtx2 = np.dot(mtx2, R.T) * s

#     # measure the dissimilarity between the two datasets
#     disparity = np.sum(np.square(mtx1 - mtx2))

#     return mtx1, mtx2, disparity, R, s

def same_stim_trial(trial_master, task_num):
    n_stim_per_ring = int(np.shape(trial_master.y)[2]-1)
    stim_rep_size = int(2*n_stim_per_ring+1)
    trial_task_num = trial_master
    trial_task_num.x[:,:,stim_rep_size:] = 0
    trial_task_num.x[:,:,stim_rep_size+task_num] = 1
    return trial_task_num

def pca_denoise(X1,X2,nD):
    pca = PCA(n_components = nD)
    X12 = np.concatenate((X1,X2),axis=1)
    _ = pca.fit_transform(X12.T)
    X1_pca = pca.transform(X1.T)
    X2_pca = pca.transform(X2.T)
    return X1_pca, X2_pca

def procrustes_fit(mtx1, mtx2):
    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s

def procrustes_test(mtx1, mtx2, R, s):
    # transform mtx2 to minimize disparity
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity

def make_procrustes_mat_stim(model_dir_all,epoch,tasks,nD = 10, batch_size = 1000):
    
    procrust = {}
    procrust['Disparity'] = np.zeros((len(tasks),len(tasks)))
    procrust['Scaling'] = np.zeros((len(tasks),len(tasks)))
    procrust['R']= np.zeros((len(tasks),len(tasks)))
    
    
    rule = 'delaygo'
    trial_all = gen_trials_from_model_dir(model_dir_all,rule, mode = 'random', batch_size = batch_size)
    trial_all_test = gen_trials_from_model_dir(model_dir_all,rule, mode = 'random', batch_size = batch_size)

    for t1_ind in range(len(tasks)):
        t1 = tasks[t1_ind]

        trial1 = same_stim_trial(trial_all, t1)
        X1,_ = gen_X_from_model_dir_epoch(model_dir_all,trial1,epoch)

        trial1_test = same_stim_trial(trial_all_test, t1)
        X1_test,_ = gen_X_from_model_dir_epoch(model_dir_all,trial1_test,epoch)

        for t2_ind in range(len(tasks)):
            if t1_ind !=t2_ind:
                t2 = tasks[t2_ind]

                trial2 = same_stim_trial(trial_all, t2)
                X2,_ = gen_X_from_model_dir_epoch(model_dir_all,trial2,epoch)
                X1_pca,X2_pca = pca_denoise(X1,X2,nD)
                prep_mtx1, prep_mtx2 = prep_procrustes(X1_pca,X2_pca)
                _, _, disparity_train, R, s = procrustes_fit(prep_mtx1, prep_mtx2)

                trial2_test = same_stim_trial(trial_all_test, t2)
                X2_test,_ = gen_X_from_model_dir_epoch(model_dir_all,trial2_test,epoch)
                X1_pca_test,X2_pca_test = pca_denoise(X1_test,X2_test,nD)
                prep_mtx1_test, prep_mtx2_test = prep_procrustes(X1_pca_test,X2_pca_test)
                mtx1, mtx2, disparity_test = procrustes_test(prep_mtx1_test, prep_mtx2_test, R, s)

                procrust['Disparity'][t1_ind,t2_ind] = disparity_test
                procrust['Scaling'][t1_ind,t2_ind] = s
                procrust['R'][t1_ind,t2_ind] = calc_R_angle(R)
    return procrust

def align_output_inds(trial_master, trial_temp):

    indices = range(np.shape(trial_master.y_loc)[1])
    n_out = np.shape(trial_master.y)[2]-1

    for ii in range(np.shape(trial_master.y_loc)[1]):
        if np.max(np.sum(abs(trial_master.x[:,ii,1:(1+n_out)]),axis = 1),axis = 0)>0:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,1:(1+n_out)]),axis = 2),axis = 0)>0
        else:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,(1+n_out):(1+2*n_out)]),axis = 2),axis = 0)>0

        loc_diff = abs(trial_temp.y_loc[-1,:]-trial_master.y_loc[-1,ii])%(2*np.pi)
        align_ind = [int(i) for i, x in enumerate(loc_diff) if x == min(loc_diff)]
        align_ind_choosey = [x for i, x in enumerate(align_ind) if ind_use[x]]
        if len(align_ind_choosey)==0:
            align_ind_choosey = align_ind
        indices[ii] = align_ind_choosey[npr.randint(len(align_ind_choosey))]
    
    trial_temp_new = trial_temp
    trial_temp_new.x = trial_temp_new.x[:,indices,:]
    trial_temp_new.y = trial_temp_new.y[:,indices,:]
    trial_temp_new.y_loc = trial_temp_new.y_loc[:,indices]
    return trial_temp_new

def project_to_output(model_dir_all,X):
    w_in, b_in, w_out, b_out = get_model_params(model_dir_all)
    y = np.dot(X.T, w_out) + b_out
    return y

def gen_mov_x(model_dir_all,rule,trial_master, batch_size = 2000, ckpt_n_dir = []):
    trial = gen_trials_from_model_dir(model_dir_all,rule,mode = 'random', batch_size = batch_size)
    trial = align_output_inds(trial_master, trial)
    _,x = gen_X_from_model_dir_epoch(model_dir_all,trial,'go1')
    x_out = project_to_output(model_dir_all,x[:,:,-1])
    err = np.sum(np.square(x_out[:,1:] - trial.y[-1,:,1:]),axis=1)
    return err, x

def make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc,trial_set = range(0,360,36)):

    fps = []
    J_xstar = []

    if (rule[:2]=='fd') & (epoch=='delay1'):
        epoch_temp = 'stim1'

        for ti in trial_set:
            filename = os.path.join(m,fp_file,rule,epoch_temp+'_'+str(round(ti,2))+'.npz')
            fp_struct = np.load(filename)
            fp_num = np.argmin(np.log10(fp_struct['qstar']))

            fps_temp = fp_struct['xstar'][fp_num,:]
            J_xstar_temp = fp_struct['J_xstar'][fp_num,:,:]

            if len(np.shape(fps_temp))==1:
                fps = fps_temp[np.newaxis,:]
                J_xstar = J_xstar_temp[np.newaxis,:,:]
            else:
                fps = np.concatenate((fps,fps_temp[np.newaxis,:]),axis = 0)
                J_xstar = np.concatenate((J_xstar,J_xstar_temp[np.newaxis,:,:]),axis = 0)

    else:
        filename = os.path.join(m,fp_file,rule,epoch+'_'+str(round(ind_stim_loc,2))+'.npz')
        fp_struct = np.load(filename)
        print(filename)
        if (epoch=='delay1') or ((rule[:2]!='fd') & (epoch=='go1')):
            fp_num = np.squeeze(np.argwhere(np.log10(fp_struct['qstar'])<-0))
        else:
            fp_num = np.argmin(np.log10(fp_struct['qstar']))

        if len(np.shape(fp_struct['xstar'][fp_num,:]))==1:
            fps = fp_struct['xstar'][fp_num,:][np.newaxis,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:][np.newaxis,:,:]
        else:
            fps = fp_struct['xstar'][fp_num,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:]
        
    return fps, J_xstar

def load_fps_J(m,fp_file,rule,epoch,ind_stim_loc,trial_set):

    ind_stim_loc_anti = (ind_stim_loc+180)%360 # ind_stim_loc is the input angle angle, anti is in the opposite direction (relevant for file names)
        
    if rule[-4:]=='anti': # anti task
        if (rule == 'delayanti') & (epoch!='stim1'): # if outside of stim epoch, inputs are the same across trials (and therefore only one set of FPs)
            ind_stim_loc_anti=180 # this is the output angle that we identified the set of fixed points on (could use any trial on this epoch)
        fps, J_xstar = make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc_anti,trial_set = trial_set) # load fixed points and Jacobian
    else: # pro task
        if (rule == 'delaygo') & (epoch!='stim1'): #again, if outside of stim epoch, inputs are the same across trials (and therefore only one set of FPs)
            ind_stim_loc=0 # this is the output angle that we identified the set of fixed points on (could use any trial on this epoch)
        fps, J_xstar = make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc,trial_set = trial_set) # load fixed points and Jacobian

    return fps, J_xstar


def make_fp_tdr_fig(m,fp_file,rule1,rule2,epoch,ind_stim_loc,tit,trial_set = range(0,360,36),dims = 'tdr'):
    
    nr = 1 # number of rows in subplots
    nc = 1 # number of columns in subplots
    ms = 10 # marker size
    
    h,trial,tasks = make_h_trial_rule(m)
    D = get_D(dims,h,trial,[rule1,],epoch,ind = -1) #identify subspace through either PCA or TDR

    fig = plt.figure(figsize=(5.5*nc,4.5*nr),tight_layout=True,facecolor='white')
    ax = plt.subplot(nr,nc,1)
    cmap=plt.get_cmap('hsv')
    
    for ind_stim_loc in trial_set:

        # load fixed points for rule 1 and plot in rule 1 axes
        fps, J_xstar = load_fps_J(m,fp_file,rule1,epoch,ind_stim_loc,trial_set)
        fp_tdr = np.dot(fps,D[rule1].T) # project FP into subspace
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'dodgerblue',markersize = ms)
        else:
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markersize = ms) # if FP diff on different trials, color by input

        # load fixed points for rule 2 and plot in rule 1 axes
        fps, J_xstar = load_fps_J(m,fp_file,rule2,epoch,ind_stim_loc,trial_set)
        fp_tdr = np.dot(fps,D[rule1].T)
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'orangered',markersize = ms)
        else:
            print(epoch)
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markerfacecolor = 'w',markersize = ms)
        
    if dims == 'tdr':
        plt.xlabel(rule1 + ' TDR input 1')
        plt.ylabel(rule1 + ' TDR input 2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Fixed Points : ' + tit)
    plt.legend((rule1,rule2))
    return ax

def make_h_combined(model_dir_all,ckpt_n_dir,tasks,trial_set,n_steps_early = 5):
    
    h_context_combined = []
    h_stim_early_combined = []
    h_stim_late_combined = []
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        rule = 'delaygo'
        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        time_set = [trial.epochs['stim1'][0]] #beginning of stim period
        n_stim_dims = np.shape(trial.x)[2]-20


        for r in range(len(tasks)):
            r_all_tasks_ind = tasks[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            # comparing Jacobians to proximity of hidden state across tasks
            # we focus on end of the context period, early, and late in the stim period
            h_context = np.reshape(h_tf[trial.epochs['stim1'][0]-1,trial_set,:],(1,-1)) # h @ end of context period
            h_stim_early = np.reshape(h_tf[trial.epochs['stim1'][0]+n_steps_early,trial_set,:],(1,-1)) # h @ 5 steps into stim
            h_stim_late = np.reshape(h_tf[trial.epochs['stim1'][1],trial_set,:],(1,-1)) # h @ end of stim period

            #concatenate activity states across tasks
            if h_context_combined == []:
                h_context_combined = h_context[np.newaxis,:]
                h_stim_late_combined = h_stim_late[np.newaxis,:]
                h_stim_early_combined = h_stim_early[np.newaxis,:]
            else:
                h_context_combined = np.concatenate((h_context_combined, h_context[np.newaxis,:]), axis=0)
                h_stim_late_combined = np.concatenate((h_stim_late_combined, h_stim_late[np.newaxis,:]), axis=0)
                h_stim_early_combined = np.concatenate((h_stim_early_combined, h_stim_early[np.newaxis,:]), axis=0)

    return h_context_combined, h_stim_late_combined, h_stim_early_combined

def generate_Beta_epoch(h_tf,trial,ind = -1,mod = 'either', ind_adjust = 0):
    Beta_epoch = {}

    for epoch in trial.epochs.keys():

        T_inds = get_T_inds(trial,epoch)
        T_use = T_inds[ind]
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        # X = h_tf[T_use,inds_use,:].T
        # X_zscore = stats.zscore(X, axis=1)
        # X_zscore_nonan = X_zscore
        # X_zscore_nonan[np.isnan(X_zscore)] = 0
        # r = X_zscore_nonan

        r = h_tf[T_use,inds_use,:].T

        if mod is 'either':
            stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
            stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        elif mod==1:
            stim1_locs = trial.stim_locs[:,0]
            stim2_locs = trial.stim_locs[:,1]
        elif mod==2:
            stim1_locs = trial.stim_locs[:,2]
            stim2_locs = trial.stim_locs[:,3]

        y_loc = trial.y_loc[-1,:]

        if epoch == 'stim1' or epoch == 'delay1':
            angle_var = stim1_locs[inds_use]
        elif epoch =='stim2' or epoch == 'delay2':
            angle_var = stim2_locs[inds_use]
        elif epoch =='go1' or epoch == 'fix1':
            angle_var = stim1_locs[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_epoch[epoch],_ = LA.qr(Beta)

        #Make sure vectors are oriented appropriately
        #first identify a trial that should be in quadrant 1
        quad1_arg = np.argmin((angle_var - np.pi/4)%(2*np.pi))
        quad1_x = h_tf[T_use,quad1_arg,:]
        dr_loc = np.dot(quad1_x,Beta_epoch[epoch])

        #flip vectors so that point is actually in quadrant 1
        if dr_loc[0]<0:
            Beta_epoch[epoch][:,0] = -Beta_epoch[epoch][:,0]
            
        if dr_loc[1]<0:
            Beta_epoch[epoch][:,1] = -Beta_epoch[epoch][:,1]

    return Beta_epoch

# def make_axes(model_dir_all,ckpt_n_dir,rule_master,epoch,ind = -1,mod = 'either'):

#     model = Model(model_dir_all)
#     with tf.Session() as sess:

#         model.saver.restore(sess,ckpt_n_dir)
#         # get all connection weights and biases as tensorflow variables
#         var_list = model.var_list
#         # evaluate the parameters after training
#         params = [sess.run(var) for var in var_list]
#         # get hparams
#         hparams = model.hp
#         trial_master = generate_trials(rule_master, hparams, mode = 'test', batch_size = 400, noise_on=False, delay_fac =1)
#         feed_dict = tools.gen_feed_dict(model, trial_master, hparams)
#         h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

#     Beta_epoch = generate_Beta_epoch(h_tf,trial_master,ind,mod = mod)
#     X_pca = Beta_epoch[epoch]    
#     D = np.concatenate((np.expand_dims(X_pca[:,0],axis=1),np.expand_dims(X_pca[:,1],axis=1)),axis = 1)
#     return D

def get_D(dims,h,trial,tasks,epoch,ind = -1):
    D = {}

    if dims=='pca':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            pca = PCA(n_components = 100)
            X = np.reshape(h[rule],(-1,N))
            _ = pca.fit_transform(X)
            D[rule] = pca.components_
    elif dims=='tdr':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            Beta_temp = generate_Beta_epoch(h[rule],trial[rule],ind = ind)
            if (rule[:2] == 'fd') & (epoch == 'delay1'):
                D[rule] = Beta_temp['stim1'].T
            else:
                D[rule] = Beta_temp[epoch].T
    return D

def get_T_inds(trial,epoch):

    T_end = trial.epochs[epoch][1] 
    if T_end is None:
        T_end = np.shape(trial.x)[0]

    T_start = trial.epochs[epoch][0]
    if T_start is None:
        T_start = 1

    T_inds = range(T_start-1,T_end)

    return T_inds

def generate_Beta_timeseries(h_tf,trial,T_inds,align_group):
    T,S,N = np.shape(h_tf)
    Beta_timeseries = np.empty((N,2,len(T_inds)))

    for t in T_inds:
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        # X = h_tf[t,inds_use,:].T
        # X_zscore = stats.zscore(X, axis=1)
        # X_zscore_nonan = X_zscore
        # X_zscore_nonan[np.isnan(X_zscore)] = 0
        # r = X_zscore_nonan
        r = h_tf[t,inds_use,:].T

        stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
        stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        y_loc = trial.y_loc[-1,:]
        
        if align_group == 'stim1':
            angle_var = stim1_locs[inds_use]
        elif align_group =='stim2':
            angle_var = stim2_locs[inds_use]
        elif align_group =='go1':
            angle_var = y_loc[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_timeseries[:,:,t],_ = LA.qr(Beta)

    return Beta_timeseries

def get_stim_cats(trial):
    #stim locations and category ids
    stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
    stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs = stim1_cats == stim2_cats
    
    return stim1_locs, stim2_locs, stim1_cats, stim2_cats

def get_Jacs(model_dir_all, ckpt_n_dir, rule_num, trial_master):

    fpf = []
    J_np = {}
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        
        trial = same_stim_trial(trial_master, rule_num)
        
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)
        
        stim1_locs, stim2_locs, stim1_cats, stim2_cats = get_stim_cats(trial)

        stim1 = trial.epochs['stim1']
        stim2 = trial.epochs['go1']

        inputs_np = []
        inputs_np.append(trial.x[stim1[0],:,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats==0,:])

        states_np = []
        states_np.append(h_tf[stim1[0]-1,:,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats==0,:])

        fpf = FixedPointFinder(model.cell,sess)
        
        for bi in range(len(states_np)):
            x_tf, F_tf = fpf._grab_RNN(states_np[bi], inputs_np[bi])
            J_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor = False)
            J_np[bi] = fpf.session.run(J_tf)
    
    return J_np

def make_h_trial_rule(model_dir_all,mode = 'random'):
    
    trial = {}
    h = {}
    
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()#model.saver.restore(sess,ckpt_n_dir)
        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hp = model.hp

        for rule in hp['rule_trains']:
            trial[rule] = generate_trials(rule, hp, mode,
                            batch_size=400, delay_fac=hp['delay_fac'],noise_on = False)

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial[rule], hp)
            h[rule] = sess.run(model.h, feed_dict=feed_dict)
            
    return h, trial, hp['rule_trains']

def plot_epoch_dynamics(m,fp_file,epoch,h,trial,rule,D_use,
                        y_set = range(0,360,36),h_epoch = [],plot_eigenspect = True,lim=4, ax_type = 'tdr',
                        epoch_axes =[],stim_loc_fp = 0):


    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)
    
    epoch_name, rule_name, epoch_axes_name, h_epoch = take_names(epoch,rule,epoch_axes = epoch_axes,
                                                                 h_epoch = h_epoch) #get intuitive names for tasks and epochs  
    nr = 1
    nc = 2
    al = .2
    
    fig = plt.figure(figsize=(4.5*nc,5*nr),tight_layout=True,facecolor='white')
    cmap = plt.get_cmap('hsv')
    
    stim1_locs = np.min(trial[rule].stim_locs[:,[0,2]],axis=1)
    y_locs = trial[rule].y_loc[-1,:]
    
    ax = plt.subplot(nr,nc,1)
    T_inds = get_T_inds(trial[rule],h_epoch)
    h_tdr = np.empty((len(T_inds),np.shape(h[rule])[1]))
    for t in range(0,np.shape(h[rule])[1],8):
        h_tdr_temp = np.dot(h[rule][T_inds,t,:],D_use)
        if stim1_locs[t]==stim_loc_fp:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,linewidth = 3)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,
                     markersize = 10,linewidth = 2)
        else:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,linewidth = 2)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,
                     markersize = 10,linewidth = 2)
        
    lim = lim
    ax = add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name)
        
    plt.title(r"$\bf{" + rule_name + "}$"+ '\n '+epoch_name+' dynamics',y = .9)
    if ax_type!='mix':
        ax.set_aspect('equal')
    
    for ind_stim_loc_anti in y_set:
        fps_anti, J_xstar = make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc_anti)
        eig_decomps = comp_eig_decomp(J_xstar)
        fps_tdr_anti = np.dot(fps_anti,D_use)
        fp_c = cmap(ind_stim_loc_anti/(360))
        plt.plot(fps_tdr_anti[:,0],fps_tdr_anti[:,1],'o',c = 'k',alpha = .5,markersize=6)
        plot_FP(fps_anti, D_use.T, eig_decomps, c='k')

        if plot_eigenspect:
            ax2 = fig.add_axes([.45, .45, .2, .2])
#             ax = plt.subplot(2*nr,2*nc,7)
            for fp_num in range(np.shape(J_xstar)[0]):
                evals, _ = LA.eig(J_xstar[fp_num,:,:]) 
                ax2.plot(evals.real,evals.imag,'.k',alpha = .3,markerfacecolor = 'k')

            ax2.plot(xs, ys,':k',linewidth = 1)
            ax2.plot(xs, -ys,':k',linewidth = 1)
            plt.xlim((.7,1.1))
            plt.ylim((-.15,.15))
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            eigenspectrum_axes(epoch,ax2)
            ax2.set_aspect('equal')   
            
    return ax

def add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name):
    if ax_type == 'out':
        out_axes(ax)
    elif ax_type == 'tdr':
        TDR_axes(epoch_axes_name,ax,rule_name)
        plt.ylim((-lim,lim))
        plt.xlim((-lim,lim))
    elif ax_type == 'mix':
        plt.ylim((-1.5,1.5))
        plt.xlim((-lim,lim))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
#         plt.xlabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+ epoch_axes_name +r' $\cos{\theta}$',fontsize = 20)
        plt.xlabel('TDR : '+r' $\sin{\theta}$',fontsize = 20)
        plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    return (ax)

def TDR_axes(epoch,ax,rule_name):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
#     plt.xlabel('TDR : '+epoch+ r' $\cos{\theta}$')
#     plt.ylabel('TDR : ' +epoch+ r' $\sin{\theta}$')
    plt.xlabel('TDR : '+r' $\cos{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    plt.ylabel('TDR : '+r' $\sin{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    
def eigenspectrum_axes(epoch,ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Real Part',fontsize = 20)
    plt.ylabel('Imaginary Part',fontsize = 20,labelpad=1)
    
def out_axes(ax):

    plt.ylim((-1.5,1.5))    
    plt.xlim((-1.5,1.5))  
    
    plt.xlabel('Output ' + r' $\cos{\theta}$',fontsize = 20)
    plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
