import os
import sys
import time
import keras

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
import utils.IOHandler as io


class ConfigFile:
    """Utility class gathering all needed information about a training session to
    ensure its reproducibility

    Attributes:
        bin_dir (str): path to bin directory
        pickle_filename (str): name format for serialized file
        checkpoints_dirname (str): checkpoints directory name
        checkpoints_format (str): format for checkpoints file names
        tensorboard_dirname (str): tensorboard logs directory name
    """

    bin_dir = os.path.join(base_dir, "bin")
    pickle_filename = "config.pickle"
    checkpoints_dirname = "checkpoints"
    checkpoints_format = "chkpt_{epoch:02d}.h5"
    tensorboard_dirname = "tensorboard"
    builder_filename = "builder.pickle"
    scores_dirname = "scores"
    observations_dirname = "observations"
    observations_subdir_format = "observations_{epoch:02d}"
    observations_format = {'src': "src_{epoch:02d}.png",
                           'tgt': "tgt_{epoch:02d}.png",
                           'pred': "pred_{epoch:02d}.png",
                           'grad_x': "grad_x_{epoch:02d}.png",
                           'grad_y': "grad_y_{epoch:02d}.png",
                           'pred_seg': "pred_seg_{epoch:02d}.png"}

    def __init__(self, session_name, input_shape=None, losses=None, loss_weights=None, optimizer=None, callbacks=None, metrics=None, epochs=None, steps_per_epoch=None, initial_epoch=0, atlas_id=None):
        """
        Args:
            session_name (str): name for the session
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
        self.session_name = session_name
        self.session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        self.input_shape = input_shape
        self.losses = losses
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.metrics = metrics
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch
        self.atlas_id = atlas_id

    def serialize(self, path=None):
        """Dumps object dictionnary as serialized pickle file
        Args:
            path (str): dumping path
        """
        if not path:
            path = os.path.join(self.session_dir, ConfigFile.pickle_filename)
        attributes = self.__dict__.copy()
        del attributes["optimizer"]
        attributes.update({"optimizer_class": self.optimizer.__class__, "optimizer_config": self.optimizer.get_config()})
        io.save_pickle(path, attributes)

    def load(self, path):
        """Loads serialized file to initalize ConfigFile instance

        Args:
            path (str): path to file
        """
        kwargs = io.load_pickle(path)
        kwargs["optimizer"] = kwargs["optimizer_class"](**kwargs["optimizer_config"])
        del kwargs["session_dir"], kwargs["optimizer_class"], kwargs["optimizer_config"]
        self.__init__(**kwargs)

    def setup_session(self, overwrite=False, timestamp=False):
        """Sets up training session directory

        Args:
            overwrite (bool): if True, overwrites existing directory (default: False)
            timestamp (bool): if True, adds timestamp to directory name (default: False)
        """
        session_name = self.session_name
        if timestamp:
            session_name = session_name + "_" + time.strftime("%Y%m%d-%H%M%S")
        io.mkdir(session_name, ConfigFile.bin_dir, overwrite)
        session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        io.mkdir(ConfigFile.checkpoints_dirname, session_dir)
        io.mkdir(ConfigFile.tensorboard_dirname, session_dir)
        io.mkdir(ConfigFile.observations_dirname, session_dir)
        io.mkdir(ConfigFile.scores_dirname, session_dir)
        ConfigFile._write_gitignore(session_dir)

    @staticmethod
    def setup_session_(session_name, overwrite=False, timestamp=False):
        """Sets up training session directory

        Args:
            session_name (str): name of session
            overwrite (bool): if True, overwrites existing directory (default: False)
            timestamp (bool): if True, adds timestamp to directory name (default: False)
        """
        if timestamp:
            session_name = session_name + "_" + time.strftime("%Y%m%d-%H%M%S")
        io.mkdir(session_name, ConfigFile.bin_dir, overwrite)
        session_dir = os.path.join(ConfigFile.bin_dir, session_name)
        io.mkdir(ConfigFile.checkpoints_dirname, session_dir)
        io.mkdir(ConfigFile.tensorboard_dirname, session_dir)
        ConfigFile._write_gitignore(session_dir)

    @staticmethod
    def _write_gitignore(dir_path):
        """Generates .gitignore file in specified directory to ignore all but
        gitignore file

        Args:
            dir_path (str): path to directory
        """
        with open(os.path.join(dir_path, ".gitignore"), "w") as f:
            f.write("*\n!.gitignore")

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def set_losses(self, losses):
        self.losses = losses

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def add_loss(self, loss, loss_weight=1.):
        self.losses.append(loss)
        self.loss_weights.append(loss_weight)

    def set_optimizer(self, optimizer):
        assert issubclass(optimizer.__class__, keras.optimizers.Optimizer), "Optimizer specified is not valid"
        self.optimizer = optimizer

    def set_callbacks(self, callbacks):
        for callback in callbacks:
            assert issubclass(callback.__class__, keras.callbacks.Callback), f"Callback {callback} is not valid"
        self.callbacks = callbacks

    def add_callback(self, callback):
        assert issubclass(callback.__class__, keras.callbacks.Callback), f"Callback {callback} is not valid"
        self.callbacks.append(callback)

    def set_atlas_id(self, atlas_id):
        self.atlas_id = atlas_id

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

    def get_atlas_id(self):
        return atlas_id
