import os
import numpy as np
import h5py
from keras.preprocessing import image
import random

base_dir = os.path.dirname(os.path.realpath(__file__))
utils_path = os.path.join(base_dir, "../../utils")
import sys
sys.path.append(utils_path)
from LungsLoader import LungsLoader
##############

SEED = 1
MNIST_DIRECTORY = os.path.join(base_dir, "../../data/3d-mnist")
DATASET_NAME = "full_dataset_vectors.h5"
ORIGIN = np.array([0., 0., 0.])
SPACING = np.array([1., 1., 1.])
TRAIN_WIDTH = 72
TRAIN_HEIGHT = 72
TRAIN_DEPTH = 72
INTENSITY = 1024


kwds_generator = {'rotation_range': 180,
                  'width_shift_range': 1.0,
                  'height_shift_range': 1.0,
                  'zoom_range': 0.5,
                  'horizontal_flip': True,
                  'vertical_flip': True}
image_gen = image.ImageDataGenerator(**kwds_generator)
loader = LungsLoader()


with h5py.File(os.path.join(MNIST_DIRECTORY, DATASET_NAME), 'r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]


def mnist_generator(n_sample,
                    seed=SEED,
                    origin=ORIGIN,
                    spacing=SPACING,
                    new_width=TRAIN_WIDTH,
                    new_height=TRAIN_HEIGHT,
                    new_depth=TRAIN_DEPTH,
                    intensity=INTENSITY):
    """
    Generator for image registration training on Mnist-3D dataset
    :param n_sample: (int) dataset size
    :param seed: (int) randomization seed
    :param origin: (np.array) origins of the ct_scan
    :param spacing: (np.array) spacing of the ct_scan
    :param new_width: (int) width to resize to
    :param new_height: (int) height to resize to
    :param new_depth: (int) depth to resize to
    :param intensity: (int) factor to rescale array's intensity
    :return: [src, tgt]: source volume and target augmented volume
    """
    vol_size = int(np.round(np.power(x_train[0].shape[0], 1 / 3)))
    vol_shape = (vol_size, vol_size, vol_size)

    zeros = np.zeros((1,) + vol_shape + (3,))
    i = 0
    np.random.seed(seed)
    while i < n_sample:
        # Select random sample
        src = random.choice(x_train).reshape(vol_shape)
        # Augment sample to generate target image
        tgt = image_gen.random_transform(src)
        # Rescale inputs to wished size
        src = loader.rescale_scan(src, origin, spacing, new_width, new_height, new_depth)[0]
        tgt = loader.rescale_scan(tgt, origin, spacing, new_width, new_height, new_depth)[0]
        # Rescale intensity
        src = intensity * src[np.newaxis, :, :, :, np.newaxis]
        tgt = intensity * tgt[np.newaxis, :, :, :, np.newaxis]
        i += 1
        yield ([src, tgt], [tgt, zeros])
