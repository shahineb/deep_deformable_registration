import os
import sys
import time
import keras
import pickle

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
import utils.IOHandler as io


class ConfigFile:

    bin_dir = os.path.join(base_dir, "bin")
    checkpoints_dirname = "checkpoints"
    checkpoints_format = "chkpt_{epoch:02d}.h5"
    tensorboard_dirname = "tensorboard"

    def __init__(self, input_shape=None, losses=None, loss_weights=None, optimizer=None, callbacks=None, metrics=None, epochs=None, steps_per_epoch=None, initial_epoch=0):
        """
        Args:
            input_shape (tuple): volume input shape
            losses (list): list of losses used for each output of the model
            reg_params (list): list of weights for the latter losses
            optimizer (keras.optimizers.Optimizer)
            callbacks (list): list of keras.callbacks.Callback
            metrics (list): list of metrics function to be tracked during training
            epochs (int): number of training epochs
            steps_per_epoch (int): number of steps to take by epoch
            initial_epoch (int): epoch number to start from (default: 0)
        """
        if losses:
            assert len(losses) == len(loss_weights), "Number of losses doesn't match number of weights precised"
        if optimizer:
            assert issubclass(optimizer.__class__, keras.optimizers.Optimizer), "Optimizer specified is not valid"
        if callbacks:
            for callback in callbacks:
                assert issubclass(callback.__class__, keras.callbacks.Callback), f"Callback {callback} is not valid"
        self.input_shape = input_shape
        self.losses = losses
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.metrics = metrics
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch

    def serialize(self, path):
        """Dumps object dictionnary as serialized pickle file
        Args:
            path (str): dumping path
        """
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def setup_session(session_name, overwrite=False, timestamp=False):
        if timestamp:
            session_name = session_name + "_" + time.strftime("%Y%m%d-%H%M%S")
        io.mkdir(session_name, ConfigFile.bin_dir, overwrite)
        session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        io.mkdir(ConfigFile.checkpoints_dirname, session_dir)
        io.mkdir(ConfigFile.tensorboard_dirname, session_dir)

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def set_losses(self, losses):
        self.losses = losses

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def set_optimizer(self, optimizer):
        assert issubclass(optimizer.__class__, keras.optimizers.Optimizer), "Optimizer specified is not valid"
        self.optimizer = optimizer

    def set_callbacks(self, callbacks):
        for callback in callbacks:
            assert issubclass(callback.__class__, keras.callbacks.Callback), f"Callback {callback} is not valid"
        self.callbacks = callbacks

    def set_metrics(self, metrics):
        self.metrics = metrics

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def set_initial_epoch(self, initial_epoch):
        self.initial_epoch = initial_epoch

    def get_input_shape(self):
        return self.input_shape

    def get_losses(self):
        return self.losses

    def get_loss_weights(self):
        return self.loss_weights

    def get_optimizer(self):
        return self.optimizer

    def get_callbacks(self):
        return self.callbacks

    def get_metrics(self):
        return self.metrics

    def get_epochs(self):
        return self.epochs

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def get_inital_epoch(self):
        return self.initial_epoch
