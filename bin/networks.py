"""
Networks model
"""

import sys
from keras.models import Model
from keras.layers import Conv3D, Input, UpSampling3D, concatenate, LeakyReLU
from keras.initializers import RandomNormal

# sys.path.append("../../../src/layers")
sys.path.append("../../../voxelmorph_review/voxelmorph/ext/neuron/neuron")
from layers import SpatialTransformer


def unet_core(vol_size, enc_nf, dec_nf):
    """
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4. e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims == 3

    # Inputs
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])

    # Down-sample path (encoder)
    x_enc = [x_in]
    for nf in enc_nf:
        x_enc.append(conv_block(x_enc[-1], nf, 2))

    # Up-sample path (decoder)
    x = x_enc[-1]
    for i, nf in enumerate(dec_nf[:-3]):
        x = conv_block(x, nf)
        x = UpSampling3D()(x)
        x = concatenate([x, x_enc[-i - 2]])

    x = conv_block(x, dec_nf[-3])
    x = conv_block(x, dec_nf[-2])

    # set back to full size
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[0]])
    x = conv_block(x, dec_nf[-1])

    return Model(inputs=[src, tgt], outputs=[x])


def voxelmorph_net(vol_size, enc_nf, dec_nf):
    ndims = len(vol_size)
    assert ndims == 3

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf)
    [src, tgt] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    flow = Conv3D(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = SpatialTransformer(interp_method='linear', indexing='ij')([src, flow])
    #  prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model


def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims == 3

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
