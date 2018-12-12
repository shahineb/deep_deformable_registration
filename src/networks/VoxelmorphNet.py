import os
import sys
from keras.models import Model
import keras.layers as KL
from keras.initializers import RandomNormal

base_dir = os.path.dirname(os.path.realpath(__file__))
neuron_path = os.path.join(base_dir, "../../../voxelmorph_review/voxelmorph/ext/neuron/neuron")
sys.path.append(neuron_path)
from layers import SpatialTransformer
from hourglassnets import HourglassNet, Unet


class VoxelmorphNet(HourglassNet):

    def __init__(self, input_shape, enc_nf, dec_nf, conv_block):
        super(VoxelmorphNet, self).__init__(input_shape)
        self.enc_nf_ = enc_nf
        self.dec_nf_ = dec_nf
        self.conv_block_ = conv_block

    def build(self):
        # Get proper convolutional layer
        conv_layer = getattr(KL, 'Conv%dD' % self.ndims_)

        # Build core Unet model
        bi_entry_unet = Unet(self.input_shape_,
                             self.enc_nf_,
                             self.dec_nf_,
                             self.conv_block_)
        unet_model = bi_entry_unet.build()
        [src, tgt] = unet_model.inputs
        x = unet_model.output

        # Transform the results into a flow field
        flow = conv_layer(self.ndims_, kernel_size=3, padding='same', name='flow',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

        # warp the source with the flow
        y = SpatialTransformer(interp_method='linear', indexing='ij')([src, flow])
        #  prepare model
        return Model(inputs=[src, tgt], outputs=[y, flow])
