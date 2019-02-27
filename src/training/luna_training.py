import os
import sys
import logging
import verboselogs
import tensorflow as tf
import pandas as pd
from keras.callbacks import TensorBoard

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
import utils.IOHandler as io
from src.training.config_file import ConfigFile
from src.generators import luna_generator as gen


class LunaTrainer:
    """Training instance for luna registration

    Args:
        model (keras.model): Model to be trained
        device (str): path to device to use for training
        config_path (str): path to serialized config file following src.training.ConfigFile
        weights_path (str): path to model weights (optional)
        use_segmentation (boolean): if true trains with segmentation data
        verbose (int): {0, 1}
        tensorboard (boolean): if true adds tensorboard callback
    """

    train_ids_filename = "train_ids.csv"
    val_ids_filename = "val_ids.csv"

    def __init__(self, model, device, config_path, weights_path=None, use_segmentation=False, verbose=1, tensorboard=False):
        self.model_ = model
        self.device_ = device
        self.main_dir_ = os.path.dirname(config_path)
        self.config = ConfigFile(session_name="")
        self.config.load(config_path)
        self.weights_path_ = weights_path
        if self.weights_path_:
            self.model_.load_weights(self.weights_path_)
        self.use_segmentation_ = use_segmentation
        self.verbose = verbose
        self.logger = verboselogs.VerboseLogger('verbose-demo')
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(verbose)
        self.tensorboard_ = tensorboard
        if self.tensorboard_:
            tb_callback = TensorBoard(log_dir=os.path.join(self.config.session_dir, ConfigFile.tensorboard_dirname),
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=True)
            self.config.add_callback(tb_callback)

    def get_config(self):
        """Returns specificities of loaded config file
        """
        return self.config.__dict__

    def fit(self, train_ids, val_ids, loop=True, shuffle=True, use_affine=True):
        """Trains model

        Args:
            train_ids (list): list of training scans ids
            val_ids (list): list of validation scans ids
            loop (boolean): If true, endlessly loop on data (default: false).
            shuffle (boolean): If true, scans are shuffled (default: false)
        """
        self.logger.verbose(f"Number of training scans : {len(train_ids)}\n")
        self.logger.verbose(f"Number of validation scans : {len(val_ids)}\n")
        pd.DataFrame(train_ids).to_csv(os.path.join(self.config.session_dir, LunaTrainer.train_ids_filename), index=False, header=False)
        pd.DataFrame(val_ids).to_csv(os.path.join(self.config.session_dir, LunaTrainer.val_ids_filename), index=False, header=False)

        (width, height, depth) = self.config.input_shape
        if self.use_segmentation_:
            train_gen = gen.scan_and_seg_generator(train_ids, width, height, depth, loop, shuffle, use_affine)
            val_gen = gen.scan_and_seg_generator(val_ids, width, height, depth, loop, shuffle, use_affine)
        else:
            train_gen = gen.scan_generator(train_ids, width, height, depth, loop, shuffle)
            val_gen = gen.scan_generator(val_ids, width, height, depth, loop, shuffle)

        self.logger.verbose("Compiling model :\n")
        self.logger.verbose(f"\t - Optimizer : {self.config.optimizer.__dict__}\n")
        self.logger.verbose(f"\t - Losses : {self.config.losses}\n")
        self.logger.verbose(f"\t - Losses weights : {self.config.loss_weights}\n")
        self.logger.verbose(f"\t - Callbacks : {self.config.callbacks}\n")
        self.model_.compile(optimizer=self.config.optimizer,
                            loss=self.config.losses,
                            loss_weights=self.config.loss_weights,
                            metrics=self.config.metrics)

        self.logger.verbose("******** Initiating training *********")
        validation_steps = int(0.2 * self.config.steps_per_epoch)
        with tf.device(self.device_):
            training_loss = self.model_.fit_generator(generator=train_gen,
                                                      initial_epoch=self.config.initial_epoch,
                                                      epochs=self.config.epochs,
                                                      callbacks=self.config.callbacks,
                                                      steps_per_epoch=self.config.steps_per_epoch,
                                                      verbose=self.verbose,
                                                      validation_data=val_gen,
                                                      validation_steps=validation_steps)

        io.save_json(os.path.join(self.main_dir_, "training_history.json"), training_loss.history)
