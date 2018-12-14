"""
keras utilities defining some default hourglass network achitectures
"""
from abc import ABCMeta, abstractmethod
from keras.models import Model
import keras.layers as kl
from keras.layers import Input, concatenate


class HourglassNet(object):
    """Super class for Hourglass networks

    Attributes:
        input_shape (tuple): network's input shape
        ndims_ (int): number of dimensions
    """
    __metaclass__ = ABCMeta

    def __init__(self, input_shape):
        self.input_shape_ = input_shape
        self.ndims_ = len(input_shape)
        assert self.ndims_ in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % self.ndims_

    @abstractmethod
    def build(self):
        pass


class Unet(HourglassNet):
    """Two entries Unet architecture

    Args:
        input_shape (tuple): network's input shape
        enc_nf (list[int]): number of filters for encoding
        dec_nf (list[int]): number of filters for decoding
        conv_block (blocks.ConvBlock): Convolutional block
    """

    def __init__(self, input_shape, enc_nf, dec_nf, conv_block):
        super(Unet, self).__init__(input_shape)
        self.enc_nf_ = enc_nf
        self.dec_nf_ = dec_nf
        self.conv_block_ = conv_block

    def build(self):
        # Set proper upsampling layer for decoding
        upsample_layer = getattr(kl, 'UpSampling%dD' % self.ndims_)

        # Inputs
        src = Input(shape=self.input_shape_ + (1,))
        tgt = Input(shape=self.input_shape_ + (1,))
        x_in = concatenate([src, tgt])

        # Encoding
        x_enc = [x_in]
        self.conv_block_.update_conv_kwargs({'strides': 2})
        for nf in self.enc_nf_:
            self.conv_block_.update_conv_kwargs({'filters': nf})
            x_enc.append(self.conv_block_.build(x_enc[-1]))

        # Decoding
        x = x_enc[-1]
        self.conv_block_.update_conv_kwargs({'strides': 1})
        for i, nf in enumerate(self.dec_nf_[:-3]):
            self.conv_block_.update_conv_kwargs({'filters': nf})
            x = self.conv_block_.build(x)
            x = upsample_layer()(x)
            x = concatenate([x, x_enc[-i - 2]])

        # Affining
        self.conv_block_.update_conv_kwargs({'filters': self.dec_nf_[-3]})
        x = self.conv_block_.build(x)
        self.conv_block_.update_conv_kwargs({'filters': self.dec_nf_[-2]})
        x = self.conv_block_.build(x)

        # Fullsize
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        self.conv_block_.update_conv_kwargs({'filters': self.dec_nf_[-1]})
        x = self.conv_block_.build(x)

        return Model(inputs=[src, tgt], outputs=[x])


class BiDecoderNet(HourglassNet):
    """
    Two entries autoencoder network architecture proposed in Shu Z, Sahasrabudhe M, Guler A, Samaras D,
    Paragios N, Kokkinos I. Deforming Autoencoders: Unsupervised Disentangling of Shape and Appearance.
    arXiv preprint arXiv:1806.06503. 2018 Jun 18.
    with parallel decoding path for linear and deformable registration
    """

    def __init__(self,
                 input_shape,
                 enc_params,
                 dec_params,
                 squeeze_block,
                 conv_block_enc,
                 conv_block_dec_deformable):
        super(BiDecoderNet, self).__init__(input_shape)
        self.enc_params_ = enc_params
        self.dec_params_ = dec_params
        self.squeeze_block_ = squeeze_block
        self.conv_block_enc_ = conv_block_enc
        self.conv_block_dec_deformable_ = conv_block_dec_deformable

    def build(self):
        # Set proper GAP layer for decoding
        globalAveragePooling_layer = getattr(kl, 'GlobalAveragePooling%dD' % self.ndims_)

        # Inputs
        src = Input(shape=self.input_shape_ + (1,))
        tgt = Input(shape=self.input_shape_ + (1,))
        x_in = concatenate([src, tgt])

        # Encoding
        x_enc = [x_in]
        for i, params in enumerate(self.enc_params_):
            self.conv_block_enc_.update(params)
            x_enc.append(self.conv_block_enc_.build(x_enc[-1]))

        x = concatenate(x_enc)

        # Deformable Decoding
        x_def = self.squeeze_block_.build(x)
        for i, params in enumerate(self.dec_params_):
            self.conv_block_dec_deformable_.update(params)
            x_def = self.conv_block_enc_.build(x_def)

        # Linear Decoding
        x_lin = globalAveragePooling_layer()(x)

        return Model(inputs=[src, tgt], outputs=[x_def, x_lin])



# TODO : to POO
#
# def maria_net(vol_size):
#     def encoding_block(input, n_filters, strides=1):
#         activation = 'LeakyReLU'
#         leakyrelu_params = {'alpha': 0.2}
#         norm_params = {'axis': len(input.get_shape()) - 2}
#         conv_params = {'filters': n_filters,
#                        'kernel_size': 3,
#                        'padding': 'same',
#                        'kernel_initializer': 'he_normal',
#                        'strides': strides}
#         x = blocks.conv_block(input,
#                               activation=activation,
#                               normalize=True,
#                               conv_kwargs=conv_params,
#                               activation_kwargs=leakyrelu_params,
#                               norm_kwargs=norm_params
#                               )
#         return x
#
#     def output_block(input, n_filters, reg_param):
#         activation = 'Sigmoid'
#         conv_params = {'filters': n_filters,
#                        'kernel_size': 3,
#                        'padding': 'same',
#                        'kernel_initializer': 'he_normal',
#                        'kernel_regularizer': l1(reg_param)}
#         x = blocks.conv_block(input,
#                               activation=activation,
#                               conv_kwargs=conv_params)
#         return x
#
#     ndims = len(vol_size)
#     assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
#     upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
#     globalAveragePooling = getattr(KL, 'GlobalAveragePooling%dD' % ndims)
#
#     # Inputs
#     src = Input(shape=vol_size + (1,))
#     tgt = Input(shape=vol_size + (1,))
#     x_in = concatenate([src, tgt])
#
#     # Encoding
#     x_enc = [x_in]
#     n_filters = [32, 64, 128, 32, 32]
#     dilatation = [1, 1, 2, 3, 5]
#     for (nf, D) in zip(n_filters, dilatation):
#         x_enc += [encoding_block(x_enc[-1], nd, D)]
#     # TODO : multi-resolution merging of all features -> ask Maria
#
#     # Decoding
#     pass
