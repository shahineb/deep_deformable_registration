import tensorflow as tf
import keras.losses as klosses


def cross_correlation(vol1, vol2):
    """Computes cross correlation between two tensors

    Args:
        vol1 (tf.Tensor)
        vol2 (tf.Tensor)
    """
    var_1 = tf.reduce_sum(tf.square(vol1 - tf.reduce_mean(vol1)))
    var_2 = tf.reduce_sum(tf.square(vol2 - tf.reduce_mean(vol2)))
    cov_12 = tf.reduce_sum((vol2 - tf.reduce_mean(vol2)) * (vol1 - tf.reduce_mean(vol1)))
    return tf.square(cov_12 / tf.sqrt(var_1 * var_2 + 1e-5))


def dice_score(seg1, seg2):
    """Computes dice loss between two segmentation tensors

    Args:
        seg1 (tf.Tensor): integer valued labels volume
        seg2 (tf.Tensor): integer valued labels volume
    """
    numerator = 2 * tf.reduce_sum(tf.cast(tf.equal(seg1, seg2), tf.int32))
    denominator = tf.size(seg1) + tf.size(seg2)
    score = numerator / denominator
    score = - tf.cast(score, tf.float32)
    return score


def registration_loss(vol1, vol2):
    return klosses.mean_squared_error(vol1, vol2) - 1. * cross_correlation(vol1, vol2)
