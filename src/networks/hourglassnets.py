"""
keras utilities defining some default hourglass network achitectures
"""
from abc import ABCMeta, abstractmethod
from keras.models import Model
import keras.layers as KL
from keras.layers import Input, concatenate


class HourglassNet(object):
    """Super class for Hourglass networks

    Attributes:
        input_shape (tuple): network's input shape
        ndims_ (int): number of dimensions
    """
    __metaclass__ = ABCMeta

    def __init__(self, input_shape):
        self.input_shape = input_shape
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
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.conv_block = conv_block

    def build(self):
        # Set proper upsampling layer for decoding
        upsample_layer = getattr(KL, 'UpSampling%dD' % self.ndims_)

        # Inputs
        src = Input(shape=self.input_shape + (1,))
        tgt = Input(shape=self.input_shape + (1,))
        x_in = concatenate([src, tgt])

        # Encoding
        x_enc = [x_in]
        self.conv_block.update_conv_kwargs({'strides': 2})
        for nf in self.enc_nf:
            self.conv_block.update_conv_kwargs({'filters': nf})
            x_enc.append(self.conv_block.build(x_enc[-1]))

        # Decoding
        x = x_enc[-1]
        self.conv_block.update_conv_kwargs({'strides': 1})
        for i, nf in enumerate(self.dec_nf[:-3]):
            self.conv_block.update_conv_kwargs({'filters': nf})
            x = self.conv_block.build(x)
            x = upsample_layer()(x)
            x = concatenate([x, x_enc[-i - 2]])

        # Affining
        self.conv_block.update_conv_kwargs({'filters': self.dec_nf[-3]})
        x = self.conv_block.build(x)
        self.conv_block.update_conv_kwargs({'filters': self.dec_nf[-2]})
        x = self.conv_block.build(x)

        # Fullsize
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        self.conv_block.update_conv_kwargs({'filters': self.dec_nf[-1]})
        x = self.conv_block.build(x)

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
                 conv_block,
                 squeeze_block):
        super(BiDecoderNet, self).__init__(input_shape)
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.squeeze_block = squeeze_block
        self.conv_block = conv_block

    def build(self):
        # Inputs
        src = Input(shape=self.input_shape + (1,))
        tgt = Input(shape=self.input_shape + (1,))
        x_in = concatenate([src, tgt])

        # Encoding
        x_enc = [x_in]
        for i, params in enumerate(self.enc_params):
            self.conv_block.update_conv_kwargs(params)
            x_enc.append(self.conv_block.build(x_enc[-1]))

        # Multiresolution feature merging
        multi_x = concatenate(x_enc)

        # Deformable Decoding
        x_def = self.squeeze_block.build(multi_x)
        self.conv_block.update_conv_kwargs({"dilation_rate": (1, 1, 1)})
        for i, params in enumerate(self.dec_params):
            self.conv_block.update_conv_kwargs(params)
            x_def = self.conv_block.build(x_def)

        return Model(inputs=[src, tgt], outputs=[x_def, multi_x])
