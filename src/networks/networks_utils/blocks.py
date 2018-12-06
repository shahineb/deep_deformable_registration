from abc import ABCMeta, abstractmethod
import keras.layers as KL
from keras import backend as K


class Block(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def get_input_dimension(inp):
        """Retrieves dimension of the data

        Args:
            inp (keras.layer)

        Returns:
            ndims (int): Dimension of the layer in [1, 2, 3]

        """
        ndims = len(inp.get_shape()) - 2
        assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
        return ndims

    @abstractmethod
    def build(self, inp):
        pass


class ConvBlock(Block):
    """Convolution + Activation + Normalization

    Attributes:
        activation (str): to choose among keras.layers activations {'ReLU', 'LeakyReLU', ...}
        normalize (bool): use BatchNormalization or not
        conv_kwargs (dict): parameters for convolution
        activation_kwargs (dict): parameters for activation
        norm_kwargs (dict): parameters for normalization
    """

    def __init__(self, activation="", normalize=False, conv_kwargs=None,
                 activation_kwargs=None, norm_kwargs=None):
        if conv_kwargs is None:
            conv_kwargs = {}
        if activation_kwargs is None:
            activation_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        self.activation_ = activation
        self.normalize_ = normalize
        self.conv_kwargs_ = conv_kwargs
        self.activation_kwargs_ = activation_kwargs
        self.norm_kwargs_ = norm_kwargs

    def update_conv_kwargs(self, params):
        self.conv_kwargs_.update(params)

    def update_activation_kwargs(self, params):
        self.activation_kwargs_.update(params)

    def update_norm_kwargs(self, params):
        self.norm_kwargs_.update(params)

    def build(self, input):
        ndims = Block.get_input_dimension(input)

        Conv = getattr(KL, 'Conv%dD' % ndims)
        x = Conv(**self.conv_kwargs_)(input)

        if self.activation_:
            Activation = getattr(KL, self.activation_)
            x = Activation(**self.activation_kwargs_)(x)
        if self.normalize_:
            x = KL.BatchNormalization(**self.norm_kwargs_)(x)
        return x


class SqueezeExciteBlock(Block):
    """Squeeze excitation block
    """

    def __init__(self, ratio):
        self.ratio_ = ratio

    def build(self, inp):
        ndims = Block.get_input_dimension(inp)

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = inp._keras_shape[channel_axis]
        se_shape = tuple([1] * ndims) + (filters,)
        global_average_pooling = getattr(KL, 'GlobalAveragePooling%dD' % ndims)

        se = global_average_pooling()(inp)
        se = KL.Reshape(se_shape)(se)
        se = KL.Dense(filters // self.ratio_, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = KL.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            permutation = (ndims,) + tuple(range(1, ndims))
            se = KL.Permute(permutation)(se)

        x = KL.multiply([inp, se])
        return x
