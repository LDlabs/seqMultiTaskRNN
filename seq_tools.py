import tensorflow as tf
import numpy as np


def compute_projection_matrices(Swh, Suh, Syy, Shh, alpha):
    # ------ rescaled eigenvalue approach ------

    # get eigendecomposition of covariance matrices
    Dwh, Vwh = np.linalg.eig(Swh)
    Duh, Vuh = np.linalg.eig(Suh)
    Dyy, Vyy = np.linalg.eig(Syy)
    Dhh, Vhh = np.linalg.eig(Shh)

    # recompute eigenvalue
    Dwh_scaled = rescale_cov_evals(Dwh, alpha)
    Duh_scaled = rescale_cov_evals(Duh, alpha)
    Dyy_scaled = rescale_cov_evals(Dyy, alpha)
    Dhh_scaled = rescale_cov_evals(Dhh, alpha)

    # reconstruct projection matrices with eigenvalues rescaled (and inverted: high variance dims are zero-ed out)
    P1 = tf.constant(np.matmul(np.matmul(Vwh, np.diag(Dwh_scaled)), Vwh.T), dtype=tf.float32)  # output space W cov(Z) W'
    P2 = tf.constant(np.matmul(np.matmul(Vuh, np.diag(Duh_scaled)), Vuh.T), dtype=tf.float32)  # input space cov(Z)
    P3 = tf.constant(np.matmul(np.matmul(Vyy, np.diag(Dyy_scaled)), Vyy.T), dtype=tf.float32)  # readiyt space  cov(Y)
    P4 = tf.constant(np.matmul(np.matmul(Vhh, np.diag(Dhh_scaled)), Vhh.T), dtype=tf.float32)  # recurrent space cov(H)

    return P1, P2, P3, P4


def rescale_cov_evals(evals, alpha):
    # ---- cut-off ----
    fvals = alpha / (alpha + evals)

    return fvals


def compute_covariance(x):
    # computes X * X.T
    return np.matmul(x, x.T) / (x.shape[1] - 1)  # or use biased estimate?
