# Load python libraries
import os
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

# Local
sys.path.append("../../../src/generators")
import mnist_generator as mnist
sys.path.append("../../../src/networks/networks_utils/")
from blocks import ConvBlock
sys.path.append("../../../src/networks/")
from VoxelmorpNets import VoxelmorphNet
sys.path.append("../../../utils")
import IOHandler as io
sys.path.append("../../../../voxelmorph_review/voxelmorph/ext/pynd-lib")
sys.path.append("../../../../voxelmorph_review/voxelmorph/ext/pytools-lib")
sys.path.append("../../../../voxelmorph_review/voxelmorph/src")
import losses

# Global variables
MODELS_DIR = "./models"
PARAMS_DIR = "./params_set"

# GPU config
gpu_id = 0
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


if __name__ == "__main__":
    for params_file in os.listdir(PARAMS_DIR):
        # Retrieve params sets
        params_path = os.path.join(PARAMS_DIR, params_file)
        params = io.load_json(params_path)
        dir_name = io.write_file_name(params)
        io.mkdir(os.path.join(MODELS_DIR, dir_name))

        # Build model
        conv_block = ConvBlock(**params['conv_block'])
        conv_block.update_conv_kwargs({"kernel_regularizer": l2(0.01)})
        voxelmorph_params = params['voxelmorph'].copy()
        voxelmorph_params['conv_block'] = conv_block
        voxelmorph_params['input_shape'] = tuple(voxelmorph_params['input_shape'])
        model = VoxelmorphNet(**voxelmorph_params)

        # Compile
        lr = params['lr']
        data_loss = params['mse']
        reg_param = params['reg_param']
        model.compile(optimizer=Adam(lr=lr),
                      loss=[data_loss, losses.Grad('l2').loss],
                      loss_weights=[1.0, reg_param])

        # Callbacks
        save_file_name = os.path.join(MODELS_DIR, dir_name, '{epoch:02d}.h5')
        save_callback = ModelCheckpoint(save_file_name)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=1e-3,
                                       patience=5,
                                       mode='min')

        # Fit
        initial_epoch = params['initial_epoch']
        nb_epochs = params['nb_epochs']
        steps_per_epoch = params['steps_per_epoch']
        validation_steps = int(0.2 * steps_per_epoch)

        size = nb_epochs * steps_per_epoch
        gen = mnist.mnist_generator(size)
        val_gen = mnist.mnist_generator(int(0.2 * size))

        with tf.device(gpu):
            model.fit_generator(gen,
                                initial_epoch=initial_epoch,
                                epochs=nb_epochs,
                                callbacks=[save_callback],
                                steps_per_epoch=steps_per_epoch,
                                verbose=1,
                                validation_data=val_gen,
                                validation_steps=validation_steps)
