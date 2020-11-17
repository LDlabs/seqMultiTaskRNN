"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import tools
from task import datasetGeneratorFromTaskDef, defineDatasetFormat
from opt_tools import AdamOptimizer_withProjection, GradientDescentOptimizer_withProjection, MomentumOptimizer_withProjection


def clip_grad(grad, max_norm):
    n = tf.norm(grad)

    # do_clip = tf.math.greater(n,max_norm)
    do_clip = 0

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        if do_clip:  # .eval():
            clipped_grad = (max_norm / n) * grad
        else:
            clipped_grad = grad

    return clipped_grad


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)


def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """

    loc = np.arctan2(y[:, 0], y[:, 1])
    return np.mod(loc, 2 * np.pi)  # check this? January 22 2019


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    loc = tf.atan2(y[:, 0], y[:, 1])
    return tf.mod(loc + np.pi, 2 * np.pi)  # check this? January 22 2019


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points

    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    corr_loc = dist < 0.1 * np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf


class LeakyRNNCell(RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.6  # 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2 / alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units
        w_in0 = (self.rng.randn(n_input, n_hidden) /
                 np.sqrt(n_input) * self._w_in_start)

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start * np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start * tools.gen_ortho_matrix(n_hidden,
                                                                rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start *
                      self.rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden))

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.constant_initializer(matrix0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            'kernel',
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._initializer)
        self._bias = self.add_variable(
            'bias',
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1 - self._alpha) * state + self._alpha * output

        return output, output


class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 hp=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        datasetType, datasetShape = defineDatasetFormat(hp)
        dataset = tf.data.Dataset.from_generator(lambda: datasetGeneratorFromTaskDef(
            hp, 'random'), datasetType, datasetShape)
        dataset = dataset.prefetch(4)
        self.datasetTensors = dataset.make_one_shot_iterator().get_next()

        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd_mom':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=hp['learning_rate'], momentum=hp['momentum'])

        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_task = tf.train.Saver(max_to_keep=100)

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = self.datasetTensors[0]  # tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        self.y = self.datasetTensors[1]  # tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        self.c_mask = self.datasetTensors[2]  # tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # self.x = tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        # self.y = tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        # self.c_mask = tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # Activation functions
        if hp['activation'] == 'power':
            def f_act(x): return tf.square(tf.nn.relu(x))
        elif hp['activation'] == 'retanh':
            def f_act(x): return tf.tanh(tf.nn.relu(x))
        elif hp['activation'] == 'relu+':
            def f_act(x): return tf.nn.relu(x + tf.constant(1.))
        else:
            f_act = getattr(tf.nn, hp['activation'])

        # Recurrent activity
        if hp['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]
            self.cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                     hp['alpha'],
                                     sigma_rec=hp['sigma_rec'],
                                     activation=hp['activation'],
                                     w_rec_init=hp['w_rec_init'],
                                     rng=self.rng)
        elif hp['rnn_type'] == 'LeakyGRU':
            self.cell = LeakyGRUCell(
                n_rnn, hp['alpha'],
                sigma_rec=hp['sigma_rec'], activation=f_act)
        elif hp['rnn_type'] == 'LSTM':
            self.cell = tf.contrib.rnn.LSTMCell(n_rnn, activation=f_act)

        elif hp['rnn_type'] == 'GRU':
            self.cell = tf.contrib.rnn.GRUCell(n_rnn, activation=f_act)
        else:
            raise NotImplementedError("""rnn_type must be one of LeakyRNN,
                    LeakyGRU, EILeakyGRU, LSTM, GRU
                    """)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            self.cell, self.x, dtype=tf.float32, time_major=True)

        # Output
        with tf.variable_scope("output"):
            # Using default initialization `glorot_uniform_initializer`
            w_out = tf.get_variable(
                'weights',
                [n_rnn, n_output],
                dtype=tf.float32
            )
            b_out = tf.get_variable(
                'biases',
                [n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32)
            )

        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat_ shape (n_time*n_batch, n_unit)
        y_hat = tf.matmul(h_shaped, w_out) + b_out

        if hp['loss_type'] == 'lsq':
            # Least-square loss
            # y_hat = tf.sigmoid(y_hat_) #removed sigmoid Jan 24, 2019
            self.cost_lsq = tf.reduce_mean(
                tf.square((y_shaped - y_hat) * self.c_mask))
        else:
            y_hat = tf.nn.softmax(y_hat_)
            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(
                self.c_mask * tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_shaped, logits=y_hat_))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    # TODO(gryang): For GRU, fix
                    self.w_rec = v[n_input:, :]
                    self.w_in = v[:n_input, :]
                else:
                    self.b_rec = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_in.shape != (n_input, n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((n_input, n_rnn)) + ', but found ' +
                             str(self.w_in.shape))

    def _build_seperate(self, hp):
        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        self.x = self.datasetTensors[0]  # tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        self.y = self.datasetTensors[1]  # tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        self.c_mask = self.datasetTensors[2]  # tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        # self.x = tf.placeholder("float", [None, None, n_input]) #add January 11 2019 #
        # self.y = tf.placeholder("float", [None, None, n_output]) #add January 11 2019 #
        # self.c_mask = tf.placeholder("float", [None, n_output])  #add January 11 2019 #

        sensory_inputs, rule_inputs = tf.split(
            self.x, [hp['rule_start'], hp['n_rule']], axis=-1)

        sensory_rnn_inputs = tf.layers.dense(sensory_inputs, n_rnn, name='sen_input')

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = tf.orthogonal_initializer()
            rule_inputs = tf.layers.dense(
                rule_inputs, hp['n_rule'], name='mix_rule',
                use_bias=False, trainable=False,
                kernel_initializer=kernel_initializer)

        rule_rnn_inputs = tf.layers.dense(rule_inputs, n_rnn, name='rule_input', use_bias=False)

        rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs

        # Recurrent activity
        self.cell = LeakyRNNCellSeparateInput(
            n_rnn, hp['alpha'],
            sigma_rec=hp['sigma_rec'],
            activation=hp['activation'],
            w_rec_init=hp['w_rec_init'],
            rng=self.rng)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            self.cell, rnn_inputs, dtype=tf.float32, time_major=True)

        # Output
        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
        # y_hat shape (n_time*n_batch, n_unit)
        y_hat = tf.layers.dense(
            h_shaped, n_output, activation=tf.nn.sigmoid, name='output')
        # Least-square loss

        self.cost_lsq = tf.reduce_mean(
            tf.square((y_shaped - y_hat) * self.c_mask))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output - 1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_rec = v
                else:
                    self.b_rec = v
            elif 'sen_input' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_sen_in = v
                else:
                    self.b_in = v
            elif 'rule_input' in v.name:
                self.w_rule = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                             str((hp['rule_start'], n_rnn)) + ', but found ' +
                             str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((hp['n_rule'], n_rnn)) + ', but found ' +
                             str(self.w_rule.shape))

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def save_ckpt(self, step):  # added Jan 9 2019
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'ckpts', 'model.ckpt' + '-' + str(step))
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        print('Variables being optimized:')
        for v in var_list:
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)

        # gradient clipping
        self.clip_max_norm = 10
        clipped_gvs = [(clip_grad(grad, self.clip_max_norm), var) for grad, var in self.grads_and_vars]
        clipped_gs = [(clip_grad(grad, self.clip_max_norm)) for grad, var in self.grads_and_vars]

        self.train_step = self.opt.apply_gradients(clipped_gvs)
        self.clipped_gs = clipped_gs  # trying to save gradients in log feb 8th

    def save_after_task(self, taskNumber):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'task_ckpts', 'model.ckpt' + '-' + str(taskNumber))
        self.saver_task.save(sess, save_path)
        print("Model saved in file: %s" % save_path)


class Sequential_Model(Model):
    """The sequential model."""

    def __init__(self, model_dir,
                 projGrad=True,
                 applyProj='both',
                 hp=None,
                 sigma_rec=None,
                 dt=None):

        self.projGrad = projGrad  # whether or not to project out interfering directions
        self.applyProj = applyProj  # how to apply update: both, left, right (for testing double-sided update rule)
        Model.__init__(self, model_dir, hp, sigma_rec, dt)

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            if self.projGrad is True:
                self.opt = AdamOptimizer_withProjection(
                    learning_rate=hp['learning_rate'], beta2=(1 - 9e-8))
            else:
                self.opt = tf.train.AdamOptimizer(
                    learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            if self.projGrad is True:
                self.opt = GradientDescentOptimizer_withProjection(
                    learning_rate=hp['learning_rate'])
            else:
                self.opt = tf.train.GradientDescentOptimizer(
                    learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd_mom':
            if self.projGrad is True:
                self.opt = MomentumOptimizer_withProjection(
                    learning_rate=hp['learning_rate'], momentum=hp['momentum'])

            else:
                self.opt = tf.train.MomentumOptimizer(
                    learning_rate=hp['learning_rate'], momentum=hp['momentum'])
        # Set cost
        self.set_optimizer()

        # Variable saver
        # self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_task = tf.train.Saver(max_to_keep=100)

    def set_optimizer(self, activity_proj=None, input_proj=None, output_proj=None, recurrent_proj=None, taskNumber=0, extra_cost=None, var_list=None, alpha=1e-3):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = self.var_list

        print('Variables being optimized:')
        for v in var_list:
            print(v)

        self.grads_and_vars = self.opt.compute_gradients(cost, var_list)

        # gradient clipping
        self.clip_max_norm = 10
        clipped_gvs = [(clip_grad(grad, self.clip_max_norm), var) for grad, var in self.grads_and_vars]
        clipped_gs = [(clip_grad(grad, self.clip_max_norm)) for grad, var in self.grads_and_vars]

        if self.projGrad is True:
            self.train_step = self.opt.apply_gradients(clipped_gvs, activity_proj, input_proj, output_proj, recurrent_proj, taskNumber)
        else:
            self.train_step = self.opt.apply_gradients(clipped_gvs)

        self.clipped_gs = clipped_gs  # trying to save gradients in log feb 8th

    def save_after_task(self, taskNumber):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'task_ckpts', 'model.ckpt' + '-' + str(taskNumber))
        self.saver_task.save(sess, save_path)
        print("Model saved in file: %s" % save_path)


class FixedPoint_Model(Model):
    """For finding fixed points."""

    def __init__(self, model_dir,
                 projGrad=True,
                 hp=None,
                 sigma_rec=0,
                 dt=None):

        Model.__init__(self, model_dir, hp, sigma_rec, dt)

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_seperate(hp)
        else:
            self._build_fused(hp)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += tf.nn.l2_loss(self.h) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * tf.add_n(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.weight_list])

        # self.saver = tf.train.Saver(self.var_list)
