import os
import sys
from keras.models import Model
from keras.regularizers import l1
import keras.layers as KL
from keras.initializers import RandomNormal

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from src.layers.diffeomorphicTransformer import diffeomorphicTransformer3D
from src.networks.hourglassnets import Unet
import utils.IOHandler as io


class DiffeomorphicUnet(Unet):
    """A Unet based architecture which output gradient flow is wrapped up
    with a diffeomorphic transforming layer
    Args:
        input_shape (tuple): (width, height, depth)
        enc_params (list): list of parameters for sequence of encoding conv blocks
        dec_params (list): list of parameters for sequence of decoding conv blocks
        conv_block (block.ConvBlock): ConvBlock instance as defined in network_utils.blocks
        flow_nf (int): Nb of filters for output flow convolutional block
    """

    def __init__(self,
                 input_shape=None,
                 enc_nf=None,
                 dec_nf=None,
                 conv_block=None,
                 flow_nf=None):
        super(DiffeomorphicUnet, self).__init__(input_shape,
                                                enc_nf,
                                                dec_nf,
                                                conv_block)
        self.flow_nf = flow_nf

    def build(self):
        # Get proper convolutional layer
        conv_layer = getattr(KL, 'Conv%dD' % self.ndims_)

        # Build core Unet architecture
        unet = Unet(self.input_shape,
                    self.enc_nf,
                    self.dec_nf,
                    self.conv_block).build()
        [src, tgt] = unet.inputs
        x = unet.output

        # Transform the results into a flow field
        flow = conv_layer(self.flow_nf,
                          kernel_size=3,
                          padding='same',
                          activation='sigmoid',
                          name='gradient_flow',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
                          kernel_regularizer=l1(1e-5))(x)

        # Wrap the source with the flow
        [deformed, displacements] = diffeomorphicTransformer3D()([src, flow])

        return Model(inputs=[src, tgt], outputs=[deformed, flow])

    def load(self, path):
        """Loads builder attributes

        Args:
            path (str): file path
        """
        kwargs = io.load_pickle(path)
        del kwargs['ndims_']
        self.__init__(**kwargs)
