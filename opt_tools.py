import tensorflow as tf


class AdamOptimizer_withProjection(tf.train.Optimizer):
    """
    implements modified version of adam optimizer
    """

    def __init__(self, learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):

        self.learning_rate = learning_rate
        # adam parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.u = {}
        self.t = tf.Variable(0.0, trainable=False)

        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, gvs, P1, P2, P3, P4, taskNumber):

        t = self.t.assign_add(1.0)

        if taskNumber == 0:
            doProj = False
        else:
            doProj = True

        update_ops = []
        for (g, v) in gvs:
            m = self.m[v].assign(self.beta1 * self.m[v] + (1 - self.beta1) * g)
            u = self.u[v].assign(self.beta2 * self.u[v] + (1 - self.beta2) * g * g)

            m_hat = m / (1 - tf.pow(self.beta1, t))
            u_hat = u / (1 - tf.pow(self.beta2, t))

            update = -self.learning_rate * m_hat / (tf.sqrt(u_hat) + self.epsilon)

            # projections are specific to recurrent or readout matrices, so check for name
            if doProj:
                if 'rnn/leaky_rnn_cell/kernel:0' in v.name:
                    # continual learning correction for recurrent/input weight update
                    update_proj = tf.matmul(tf.matmul(P2, update), P1)
                elif 'output/weights:0' in v.name:
                    # continual learning correction for readout weight update
                    update_proj = tf.matmul(tf.matmul(P4, update), P3)
                    # update_proj = tf.matmul(P1, update)

            else:
                update_proj = update

            update_ops.append(v.assign_add(update_proj))

        return tf.group(*update_ops)


class GradientDescentOptimizer_withProjection(tf.train.Optimizer):
    """
    implements modified version of SGD optimizer
    """

    def __init__(self, learning_rate=0.001):

        self.learning_rate = learning_rate

    def apply_gradients(self, gvs, P1, P2, P3, P4, taskNumber):

        if taskNumber == 0:
            doProj = False
        else:
            doProj = True

        update_ops = []
        for (g, v) in gvs:

            update = -self.learning_rate * g

            if doProj:
                if 'rnn/leaky_rnn_cell/kernel:0' in v.name:
                    # continual learning correction for recurrent/input weight update
                    update_proj = tf.matmul(tf.matmul(P2, update), P1)
                elif 'output/weights:0' in v.name:
                    # continual learning correction for readout weight update
                    update_proj = tf.matmul(tf.matmul(P4, update), P3)
                    # update_proj = tf.matmul(P1, update)
            else:
                update_proj = update

            update_ops.append(v.assign_add(update_proj))

        return tf.group(*update_ops)


class MomentumOptimizer_withProjection(tf.train.Optimizer):
    """
    implements modified version of SGD optimizer
    """

    def __init__(self, learning_rate=0.001,
                 momentum=0.1):

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.m = {}
        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, gvs, P1, P2, P3, P4, taskNumber):

        if taskNumber == 0:
            doProj = False
        else:
            doProj = True

        update_ops = []
        for (g, v) in gvs:
            self.m[v] = self.momentum * self.m[v] + g

            update = -self.learning_rate * self.m[v]

            if doProj:
                if 'rnn/leaky_rnn_cell/kernel:0' in v.name:
                    # continual learning correction for recurrent/input weight update
                    update_proj = tf.matmul(tf.matmul(P2, update), P1)
                elif 'output/weights:0' in v.name:
                    # continual learning correction for readout weight update
                    # update_proj = tf.matmul(P1, update)
                    update_proj = tf.matmul(tf.matmul(P4, update), P3)
            else:
                update_proj = update

            update_ops.append(v.assign_add(update_proj))

        return tf.group(*update_ops)
