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


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def plot_vol(cube, angle=0):
    def normalize(cube):
        max_val = np.max(cube)
        min_val = np.min(cube)
        cube = (cube - min_val) / (max_val - min_val)
        return cube
    vol_size = int(np.round(np.power(x_train[0].shape[0], 1 / 3)))
    cube = normalize(cube)

    facecolors = plt.cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(20 / 2.54, 18 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=vol_size * 2)
    ax.set_ylim(top=vol_size * 2)
    ax.set_zlim(top=vol_size * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.tight_layout()
    plt.show()
