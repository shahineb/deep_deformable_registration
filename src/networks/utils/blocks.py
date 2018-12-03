"""
keras utilities defining standard block architecures
"""
import keras.layers as KL
from keras import backend as K


def conv_block(input, activation=None, normalization=False, conv_kwargs={}, activation_kwargs={}, norm_kwargs={}):
    """convolution block

    Args:
        input (keras.layer): layer to add conv_block on top of
        activation (str): to choose among keras.layers activations {'ReLU', 'LeakyReLU', ...}
        normalization (bool): use BatchNormalization or not
        conv_kwargs (dict): parameters for convolution
        activation_kwargs (dict): parameters for activation
        norm_kwargs (dict): parameters for normalization
    """
    ndims = len(input.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x = Conv(**conv_kwargs)(input)

    if activation:
        Activation = getattr(KL, activation)
        x = Activation(**activation_kwargs)(x)
    if normalization:
        x = KL.BatchNormalization(**norm_kwargs)(x)
    return x


def squeeze_excite_block(input, ratio=16):
    """Squeeze excitation block
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input._keras_shape[channel_axis]
    ndims = len(input.get_shape()) - 2
    se_shape = tuple([1] * ndims) + (filters,)
    globalAveragePooling = getattr(KL, 'GlobalAveragePooling%dD' % ndims)

    se = globalAveragePooling()(input)
    se = KL.Reshape(se_shape)(se)
    se = KL.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = KL.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        permutation = (ndims,) + tuple(range(1, ndims))
        se = KL.Permute(permutation)(se)

    x = KL.multiply([input, se])
    return x
