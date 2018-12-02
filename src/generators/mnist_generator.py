import os
import numpy as np
import h5py
from keras.preprocessing import image
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    vol_size = int(np.round(np.power(x_train[0].shape[0], 1 / 3)))
    vol_shape = (vol_size, vol_size, vol_size)

    zeros = np.zeros((1,) + vol_shape + (3,))
    i = 0
    np.random.seed(seed)
    while i < n_sample:
        src = random.choice(x_train).reshape(vol_shape)
        tgt = image_gen.random_transform(src)
        src = src[np.newaxis, :, :, :, np.newaxis]
        tgt = tgt[np.newaxis, :, :, :, np.newaxis]
        i += 1
        yield ([src, tgt], [tgt, zeros])


def plot_vol(volume):
    # TODO : shouldnt be here, to be set in propoer plotting framework
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(volume, edgecolor='k')
    plt.show()
