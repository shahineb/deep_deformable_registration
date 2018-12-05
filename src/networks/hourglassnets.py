"""
keras utilities defining some default hourglass network achitectures
"""
from keras.models import Model
import keras.layers as KL
from keras.regularizers import l1, l2
from keras.layers import Input, concatenate
import blocks.blocks as blocks


def unet(vol_size, enc_nf, dec_nf):
    """
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters (1, n_blocks)
    :param dec_nf: list of decoder filters. (1, n_blocks + 2)
    :return: the keras model
    """
    def unet_conv_block(input, n_filters, strides=1):
        activation = 'LeakyReLU'
        normalize = True
        leakyrelu_params = {'alpha': 0.2}
        conv_params = {'filters': n_filters,
                       'kernel_size': 3,
                       'padding': 'same',
                       'kernel_initializer': 'he_normal',
                       'kernel_regularizer': l2(0.01),
                       'strides': strides}
        x = blocks.conv_block(input,
                              activation=activation,
                              normalization=normalize,
                              conv_kwargs=conv_params,
                              activation_kwargs=leakyrelu_params
                              )
        return x

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # Inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # Encoding
    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(unet_conv_block(x_enc[-1], nf, 2))

    # Decoding
    x = x_enc[-1]
    for i, nf in enumerate(dec_nf[:-3]):
        x = unet_conv_block(x, nf)
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[-i - 2]])

    x = unet_conv_block(x, dec_nf[-3])
    x = unet_conv_block(x, dec_nf[-2])

    # Fullsize
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[0]])
    x = unet_conv_block(x, dec_nf[-1])

    return Model(inputs=[src, tgt], outputs=[x])


def maria_net(vol_size):
    def encoding_block(input, n_filters, strides=1):
        activation = 'LeakyReLU'
        leakyrelu_params = {'alpha': 0.2}
        norm_params = {'axis': len(input.get_shape()) - 2}
        conv_params = {'filters': n_filters,
                       'kernel_size': 3,
                       'padding': 'same',
                       'kernel_initializer': 'he_normal',
                       'strides': strides}
        x = blocks.conv_block(input,
                              activation=activation,
                              normalize=True,
                              conv_kwargs=conv_params,
                              activation_kwargs=leakyrelu_params,
                              norm_kwargs=norm_params
                              )
        return x

    def output_block(input, n_filters, reg_param):
        activation = 'Sigmoid'
        conv_params = {'filters': n_filters,
                       'kernel_size': 3,
                       'padding': 'same',
                       'kernel_initializer': 'he_normal',
                       'kernel_regularizer': l1(reg_param)}
        x = blocks.conv_block(input,
                              activation=activation,
                              conv_kwargs=conv_params)
        return x

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    globalAveragePooling = getattr(KL, 'GlobalAveragePooling%dD' % ndims)

    # Inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # Encoding
    x_enc = [x_in]
    n_filters = [32, 64, 128, 32, 32]
    dilatation = [1, 1, 2, 3, 5]
    for (nf, D) in zip(n_filters, dilatation):
        x_enc += [encoding_block(x_enc[-1], nd, D)]
    # TODO : multi-resolution merging of all features -> ask Maria

    # Decoding
    pass
