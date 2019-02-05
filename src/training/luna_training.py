import os
import sys
import logging
import verboselogs
import tensorflow as tf

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
import utils.IOHandler as io
from src.training.config_file import ConfigFile
from src.generators import luna_generator as gen
from src.evaluation.Testing_pipeline import pipeline_test_set


class LunaTrainer:
    """Training instance for luna registration

    Args:
        model (keras.model): Model to be trained
        device (str): path to device to use for training
        config_path (str): path to serialized config file following src.training.ConfigFile
        weights_path (str): path to model weights (optional)
        use_segmentation (boolean): if true trains with segmentation data
        verbose (int): {0, 1}
    """

    def __init__(self, model, device, config_path, weights_path=None, use_segmentation=False, verbose=1):
        self.model_ = model
        self.device_ = device
        self.main_dir_ = os.path.dirname(config_path)
        self.config = ConfigFile(**io.load_pickle(config_path))
        self.weights_path_ = weights_path
        if self.weights_path_:
            self.model_.load_weights(self.weights_path_)
        self.use_segmentation_ = use_segmentation
        self.verbose = verbose
        self.logger = verboselogs.VerboseLogger('verbose-demo')
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(verbose)

    def show_config(self):
        """Prints specificities of loaded config file
        """
        print(self.config.__dict__)

    def fit(self, train_ids, val_ids):
        """Trains model

        Args:
            train_ids (list): list of training scans ids
            val_ids (list): list of validation scans ids
        """
        self.logger.verbose(f"Number of training scans : {len(train_ids)}\n")
        self.logger.verbose(f"Number of validation scans : {len(val_ids)}\n")
        (width, height, depth) = self.config.input_shape
        if self.use_segmentation_:
            train_gen = gen.scan_and_seg_generator(train_ids, width, height, depth, loop=True)
            val_gen = gen.scan_and_seg_generator(val_ids, width, height, depth, loop=True)
        else:
            train_gen = gen.scan_generator(train_ids, width, height, depth, loop=True)
            val_gen = gen.scan_generator(val_ids, width, height, depth, loop=True)

        self.logger.verbose("Compiling model :\n")
        self.logger.verbose(f"\t - Optimizer : {self.config.optimizer.__dict__}\n")
        self.logger.verbose(f"\t - Losses : {self.config.losses}\n")
        self.logger.verbose(f"\t - Losses weights : {self.config.loss_weights}\n")
        self.model.compile(optimizer=self.config.optimizer,
                           loss=self.config.losses,
                           loss_weights=self.config.loss_weights,
                           metrics=self.config.metrics)

        self.logger.verbose("******** Initiating training *********")
        validation_steps = int(0.2 * self.config.steps_per_epoch)
        with tf.device(self.device):
            training_loss = self.model.fit_generator(generator=train_gen,
                                                     initial_epoch=self.config.initial_epoch,
                                                     epochs=self.config.epochs,
                                                     callbacks=self.config.callbacks,
                                                     steps_per_epoch=self.config.steps_per_epoch,
                                                     verbose=self.verbose,
                                                     validation_data=val_gen,
                                                     validation_steps=validation_steps)

        io.save_json(os.path.join(self.main_dir, "training_history.json"), training_loss.history)

    def test(self, val_gen=None, model_name='model'):
        """
        :param val_gen: generator for the validation set
        :param model_name: name of the model (filename)
        """
        if val_gen is not None:
            self.val_gen = val_gen
        pipeline_test_set(self.model, self.val_gen, model_name, self.segmentation)
        return None
