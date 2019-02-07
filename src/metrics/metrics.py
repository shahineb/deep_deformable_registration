import numpy as np
import tensorflow as tf


def dice(vol1, vol2, labels=None, nargout=1):
    """
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    :param vol1: nd array. The first volume (e.g. predicted volume)
    :param vol2: nd array. The second volume (e.g. "true" volume)
    :param labels: optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    :param nargout: optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)
    :return: if nargout == 1 : dice : vector of dice measures for each labels
             if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
             dice was computed
    """
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return dicem, labels


def cross_correlation(vol1, vol2):
    # TODO : double check cross correlation computation
    """Computes cross correlation between two arrays

    Args:
        vol1 (tf.tensor)
        vol2 (tf.tensor)
    """
    var_1 = tf.reduce_sum(tf.square(vol1 - tf.reduce_mean(vol1)))
    var_2 = tf.reduce_sum(tf.square(vol2 - tf.reduce_mean(vol2)))
    cov_12 = tf.reduce_sum((vol2 - tf.reduce_mean(vol2)) * (vol1 - tf.reduce_mean(vol1)))
    return tf.square(cov_12 / tf.sqrt(var_1 * var_2 + 0.00001))
