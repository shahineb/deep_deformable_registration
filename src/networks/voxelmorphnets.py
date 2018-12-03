import sys
from keras.models import Model
from keras.layers import Conv3D
from keras.initializers import RandomNormal

sys.path.append("../../../voxelmorph_review/voxelmorph/ext/neuron/neuron")
from layers import SpatialTransformer
from hourglassnets import unet


def voxelmorph_net(vol_size, enc_nf, dec_nf):
    """Unet based architecture proposed at CVPR by voxelmorph team

    Args:
        vol_size (type): Description of parameter `vol_size`.
        enc_nf (type): Description of parameter `enc_nf`.
        dec_nf (type): Description of parameter `dec_nf`.

    Returns:
        type: Description of returned object.

    """
    ndims = len(vol_size)
    assert ndims == 3

    # get the core model
    unet_model = unet(vol_size, enc_nf, dec_nf)
    [src, tgt] = unet_model.inputs
    x = unet_model.output

    # Transform the results into a flow field
    flow = Conv3D(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = SpatialTransformer(interp_method='linear', indexing='ij')([src, flow])
    #  prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model
