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
loader = LungsLoader()
handler = ScanHandler(plt)


class OutputObserver(Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, src_filepath, tgt_filepath, pred_filepath, grad_x_filepath, grad_y_filepath,
                 src_id=None, tgt_id=None, random_seed=777):
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
        src = loader.get_scan(self.src_id)[0][np.newaxis, :, :, :, np.newaxis]
        tgt = loader.get_scan(self.src_id)[0][np.newaxis, :, :, :, np.newaxis]
        output = self.model.predict([src, tgt])
        fig, _ = handler.display_n_slices(src.squeeze(), n=4, return_fig=True)
        fig.savefig(self.src_filepath)
        fig, _ = handler.display_n_slices(tgt.squeeze(), n=4, save_path=self.tgt_filepath)
        fig.savefig(self.tgt_filepath)
        fig, _ = handler.display_n_slices(output[0].squeeze(), n=4, save_path=self.pred_filepath)
        fig.savefig(self.pred_filepath)
        fig, _ = handler.display_n_slices(output[1][:, :, :, 0].squeeze(), n=4, save_path=self.grad_x_filepath)
        fig.savefig(self.grad_x_filepath)
        fig, _ = handler.display_n_slices(output[1][:, :, :, 1].squeeze(), n=4, save_path=self.grad_y_filepath)
        fig.savefig(self.grad_y_filepath)
