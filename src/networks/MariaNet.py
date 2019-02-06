import os
import sys
import pickle
from keras.models import Model
from keras.regularizers import l1
import keras.layers as KL
from keras.initializers import RandomNormal

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from src.layers.diffeomorphicTransformer import diffeomorphicTransformer3D
from src.networks.hourglassnets import BiDecoderNet


class MariaNet(BiDecoderNet):

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
        self.def_flow_nf_ = def_flow_nf
        self.lin_flow_nf_ = lin_flow_nf

    def build(self):
        # Get proper convolutional layer
        conv_layer = getattr(KL, 'Conv%dD' % self.ndims_)

        # Build core architecture
        bi_decoding_net = super(MariaNet, self).build()
        [src, tgt] = bi_decoding_net.inputs
        [x_def, x_lin] = bi_decoding_net.output

        # Transform the results into a flow field
        deformable_grad_flow = conv_layer(self.def_flow_nf_,
                                          kernel_size=3,
                                          padding='same',
                                          activation='sigmoid',
                                          name='deformable_gradient_flow',
                                          kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
                                          kernel_regularizer=l1(1e-5))(x_def)
        linear_flow = conv_layer(self.lin_flow_nf_,
                                 kernel_size=3,
                                 padding='same',
                                 activation='linear',
                                 name='linear_flow',
                                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),
                                 kernel_regularizer=l1(1e-5))(x_def)

        # Wrap the source with the flow
        [deformed, displacements] = diffeomorphicTransformer3D()([src, deformable_grad_flow, linear_flow])

        #  prepare model
        return Model(inputs=[src, tgt], outputs=[deformed, deformable_grad_flow, linear_flow])

    def serialize(self, path):
        """Dumps object dictionnary as serialized pickle file
        Args:
            path (str): dumping path
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
