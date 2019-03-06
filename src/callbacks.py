import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from utils.LungsLoader import LungsLoader
from utils.ScanHandler import ScanHandler
import utils.IOHandler as io
from src.training.config_file import ConfigFile

loader = LungsLoader()
handler = ScanHandler(plt)


class OutputObserver(Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, session_dir, input_shape, src_filepath, tgt_filepath, pred_filepath, grad_x_filepath, grad_y_filepath,
                 src_id=None, tgt_id=None, random_seed=13):
        self.session_dir = session_dir
        self.input_shape = input_shape
        self.src_filepath = src_filepath
        self.tgt_filepath = tgt_filepath
        self.pred_filepath = pred_filepath
        self.grad_x_filepath = grad_x_filepath
        self.grad_y_filepath = grad_y_filepath
        self.random_seed_ = random_seed
        random.seed(random_seed)
        self.src_id = src_id or random.choice(loader.get_scan_ids())
        self.tgt_id = tgt_id or random.choice(loader.get_scan_ids())

    def on_epoch_end(self, epoch, logs={}):
        observations_subdir_format = ConfigFile.observations_subdir_format.format(epoch=epoch + 1, **logs)
        src_filepath = self.src_filepath.format(epoch=epoch + 1, **logs)
        tgt_filepath = self.tgt_filepath.format(epoch=epoch + 1, **logs)
        pred_filepath = self.pred_filepath.format(epoch=epoch + 1, **logs)
        grad_x_filepath = self.grad_x_filepath.format(epoch=epoch + 1, **logs)
        grad_y_filepath = self.grad_y_filepath.format(epoch=epoch + 1, **logs)
        
        observations_dir = os.path.join(self.session_dir, ConfigFile.observations_dirname)
        io.mkdir(observations_subdir_format, observations_dir)
        
        src_gen = loader.preprocess_scans([self.src_id], *self.input_shape)
        tgt_gen = loader.preprocess_scans([self.tgt_id], *self.input_shape)
        src = next(src_gen)[0][np.newaxis, :, :, :, np.newaxis]
        tgt = next(tgt_gen)[0][np.newaxis, :, :, :, np.newaxis]
        output = self.model.predict([src, tgt])
        
        fig, _ = handler.display_n_slices(src.squeeze(), n=4, return_fig=True)
        fig.savefig(os.path.join(observations_dir, observations_subdir_format, src_filepath))
        plt.close()
        fig, _ = handler.display_n_slices(tgt.squeeze(), n=4, return_fig=True)
        fig.savefig(os.path.join(observations_dir, observations_subdir_format, tgt_filepath))
        plt.close()
        fig, _ = handler.display_n_slices(output[0].squeeze(), n=4, return_fig=True)
        fig.savefig(os.path.join(observations_dir, observations_subdir_format, pred_filepath))
        plt.close()
        fig, _ = handler.display_n_slices(output[1].squeeze()[:, :, :, 0], n=4, return_fig=True)
        fig.savefig(os.path.join(observations_dir, observations_subdir_format, grad_x_filepath))
        plt.close()
        fig, _ = handler.display_n_slices(output[1].squeeze()[:, :, :, 1], n=4, return_fig=True)
        fig.savefig(os.path.join(observations_dir, observations_subdir_format, grad_y_filepath))
        plt.close()
