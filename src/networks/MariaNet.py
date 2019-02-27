import os
import sys
from keras.models import Model
from keras.regularizers import l1
import keras.layers as KL
from keras.initializers import RandomNormal

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from src.layers.diffeomorphicTransformer import diffeomorphicTransformer3D
from src.networks.hourglassnets import BiDecoderNet
import utils.IOHandler as io


class MariaNet(BiDecoderNet):
    """Implementation of registration network proposed in
    Christodoulis et al. 2018 (https://arxiv.org/abs/1809.06226)

    Args:
        input_shape (tuple): (width, height, depth)
        enc_params (list): list of parameters for sequence of encoding conv blocks
        dec_params (list): list of parameters for sequence of decoding conv blocks
        conv_block (block.ConvBlock): ConvBlock instance as defined in network_utils.blocks
        squeeze_block (block.SqueezeExciteBlock): SqueezeExciteBlock instance as defined in network_utils.blocks
        def_flow_nf (int): Nb of filters for deformable flow convolutional block
        lin_flow_nf (int): Nb of filters for linear flow convolutional block
    """

    def __init__(self,
                 input_shape,
                 enc_params,
                 dec_params,
                 conv_block,
                 squeeze_block,
                 def_flow_nf,
                 lin_flow_nf):
        super(MariaNet, self).__init__(input_shape,
                                       enc_params,
                                       dec_params,
                                       conv_block,
                                       squeeze_block)
        self.def_flow_nf = def_flow_nf
        self.lin_flow_nf = lin_flow_nf

    def build(self):
        # Get proper convolutional layer and GAP layer for decoding
        conv_layer = getattr(KL, 'Conv%dD' % self.ndims_)
        globalAveragePooling_layer = getattr(KL, 'GlobalAveragePooling%dD' % self.ndims_)

        # Build core architecture
        bi_decoding_net = super(MariaNet, self).build()
        [src, tgt] = bi_decoding_net.inputs
        [x_def, multi_x] = bi_decoding_net.output

        # Transform the results into a flow field
        deformable_grad_flow = conv_layer(self.def_flow_nf,
                                          kernel_size=3,
                                          padding='same',
                                          activation='sigmoid',
                                          name='deformable_gradient_flow',
                                          kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
                                          kernel_regularizer=l1(1e-5))(x_def)
        linear_flow = conv_layer(self.lin_flow_nf,
                                 kernel_size=3,
                                 padding='same',
                                 activation='linear',
                                 name='linear_flow',
                                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
                                 kernel_regularizer=l1(1e-5))(multi_x)
        linear_flow = globalAveragePooling_layer(name="averaged_linear_flow")(linear_flow)

        # Wrap the source with the flow
        [deformed, displacements] = diffeomorphicTransformer3D()([src, deformable_grad_flow, linear_flow])

        #  prepare model
        return Model(inputs=[src, tgt], outputs=[deformed, deformable_grad_flow, linear_flow])

    def load(self, path):
        """Loads builder attributes

        Args:
            path (str): file path
        """
        kwargs = io.load_pickle(path)
        del kwargs['ndims_']
        self.__init__(**kwargs)
