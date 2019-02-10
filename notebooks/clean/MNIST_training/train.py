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

utils_path = os.path.join(base_dir, "../../../utils")
generators_path = os.path.join(base_dir, "../../../src/generators")
networks_utils_path = os.path.join(base_dir, "../../../src/networks/networks_utils")
networks_path = os.path.join(base_dir, "../../../src/networks")
neuron_path = os.path.join(base_dir, "../../../../voxelmorph_review/voxelmorph/ext/neuron/neuron")
pynd_path = os.path.join(base_dir, "../../../../voxelmorph_review/voxelmorph/ext/pynd-lib")
pytools_path = os.path.join(base_dir, "../../../../voxelmorph_review/voxelmorph/ext/pytools-lib")
training_tools_path = os.path.join(base_dir, "../../../src/training")
# TODO : change loss path to thera_reg_oma/src/training
src_path = os.path.join(base_dir, "../../../../voxelmorph_review/voxelmorph/src")

sys.path.append(utils_path)
sys.path.append(generators_path)
sys.path.append(networks_utils_path)
sys.path.append(networks_path)
sys.path.append(neuron_path)
sys.path.append(pynd_path)
sys.path.append(pytools_path)
sys.path.append(training_tools_path)
sys.path.append(src_path)
from blocks import ConvBlock
from VoxelmorphNet import VoxelmorphNet
import IOHandler as io
import mnist_generator as mnist
import losses
from training import temp_loss_monitoring as tmp

# Global variables
MODELS_DIR = os.path.join(base_dir, "models")
PARAMS_DIR = os.path.join(base_dir, "params_set")

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
        # Retrieve params
        params_path = os.path.join(PARAMS_DIR, params_file)
        params = io.load_json(params_path)

        # Create current saving directories
        dir_name = params_file.split(".")[0]
        io.mkdir(dir_name=dir_name, location=MODELS_DIR)
        local_dir = os.path.join(MODELS_DIR, dir_name)
        io.mkdir(dir_name="models", location=local_dir)

        # Build model
        conv_block = ConvBlock(**params['conv_block'])
        conv_block.update_conv_kwargs({"kernel_regularizer": l2(0.01)})

        voxelmorph_params = params['voxelmorph'].copy()
        voxelmorph_params['conv_block'] = conv_block
        voxelmorph_params['input_shape'] = tuple(voxelmorph_params['input_shape'])
        model = VoxelmorphNet(**voxelmorph_params).build()

        # Compile
        lr = params['lr']
        data_loss = params['data_loss']
        reg_param = params['reg_param']
        model.compile(optimizer=Adam(lr=lr),
                      loss=[data_loss, losses.Grad('l2').loss],
                      loss_weights=[1.0, reg_param])

        # Callbacks
        save_file_name = os.path.join(MODELS_DIR, dir_name, "models", '{epoch:02d}.h5')
        save_callback = ModelCheckpoint(save_file_name,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)
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
            logs = model.fit_generator(gen,
                                       initial_epoch=initial_epoch,
                                       epochs=nb_epochs,
                                       callbacks=[save_callback],
                                       steps_per_epoch=steps_per_epoch,
                                       verbose=1,
                                       validation_data=val_gen,
                                       validation_steps=validation_steps)

        # Record loss
        dump_path = os.path.join(local_dir, "loss_history.png")
        tmp.plot_loss(logs_history=logs.history, dump_path=dump_path)
