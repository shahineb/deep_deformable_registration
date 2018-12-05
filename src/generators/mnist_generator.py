import os
import numpy as np
import h5py
from keras.preprocessing import image
from keras.utils import normalize
import random

## TO EDIT
import sys
sys.path.append("../../utils")
from LungsLoader import LungsLoader
loader = LungsLoader()
origin = np.array([0., 0., 0.])
spacing = np.array([1., 1., 1.])
new_width = 72
new_height = 72
new_depth = 72
##############

SEED = 1
MNIST_DIRECTORY = "../../data/3d-mnist"
DATASET_NAME = "full_dataset_vectors.h5"

kwds_generator = {'rotation_range': 180,
                  'width_shift_range': 1.0,
                  'height_shift_range': 1.0,
                  'zoom_range': 0.5,
                  'horizontal_flip': True,
                  'vertical_flip': True}
image_gen = image.ImageDataGenerator(**kwds_generator)


with h5py.File(os.path.join(MNIST_DIRECTORY, DATASET_NAME), 'r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]


def mnist_generator(n_sample, seed=SEED):
    """Generator for image registration training on Mnist-3D dataset

    Args:
        n_sample (int): dataset size
        seed (int): randomization seed

    Returns:
        [src, tgt]: source volume and target augmented volume
    """
    vol_size = int(np.round(np.power(x_train[0].shape[0], 1 / 3)))
    vol_shape = (vol_size, vol_size, vol_size)

    zeros = np.zeros((1,) + vol_shape + (3,))
    i = 0
    np.random.seed(seed)
    while i < n_sample:
        src = random.choice(x_train).reshape(vol_shape)
        tgt = image_gen.random_transform(src)
        src = loader._rescale_scan(src, origin, spacing, new_width, new_height, new_depth)[0]
        tgt = loader._rescale_scan(tgt, origin, spacing, new_width, new_height, new_depth)[0]
        src = 1024 * src[np.newaxis, :, :, :, np.newaxis]
        tgt = 1024 * tgt[np.newaxis, :, :, :, np.newaxis]
        i += 1
        yield ([src, tgt], [tgt, zeros])
