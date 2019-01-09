# Load python libraries
import os
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

# Local imports
base_dir = os.path.dirname(os.path.realpath(__file__))
