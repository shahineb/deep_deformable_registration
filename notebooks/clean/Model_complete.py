import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import json


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
utils_dir = os.path.join(base_dir, 'thera_reg_oma/utils')
losses_dir = os.path.join(base_dir, 'voxelmorph_review/voxelmorph/src')
pipe_dir = os.path.join(base_dir, 'thera_reg_oma/notebooks/clean')
sys.path.append(utils_dir)
sys.path.append(losses_dir)
from LungsLoader import LungsLoader
import losses
from Testing_pipeline import pipeline_test_set

loader = LungsLoader()
gpu_id = 1
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


def scan_generator(scans_ids, width, height, depth, loop=False):
    preprocess_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop)
    while True:
        src_scan = next(preprocess_gen)[0]
        tgt_scan = next(preprocess_gen)[0]
        src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
        tgt_scan = tgt_scan[np.newaxis, :, :, :, np.newaxis]
        zeros = np.zeros((1,) + (width, height, depth) + (3,))
        yield ([src_scan, tgt_scan], [tgt_scan, zeros])


def scan_and_seg_generator(scans_ids, width, height, depth, loop=False):
    global loader
    scans_ids = iter(scans_ids)

    while True:
        src_id = next(scans_ids)
        tgt_id = next(scans_ids)

        scan_gen = loader.preprocess_scans([src_id, tgt_id], width, height, depth)
        seg_gen = loader.preprocess_segmentations([src_id, tgt_id], width, height, depth)

        src_scan = next(scan_gen)[0]
        tgt_scan = next(scan_gen)[0]
        src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
        tgt_scan = tgt_scan[np.newaxis, :, :, :, np.newaxis]
        zeros = np.zeros((1,) + (width, height, depth) + (3,))

        src_seg = next(seg_gen)[0]
        tgt_seg = next(seg_gen)[0]
        src_seg = src_seg[np.newaxis, :, :, :, np.newaxis]
        tgt_seg = tgt_seg[np.newaxis, :, :, :, np.newaxis]

        yield ([src_scan, tgt_scan, src_seg], [tgt_scan, zeros, tgt_seg])


class Model:
    """
    params = dict of parameters including training
    Training
    Testing

    """
    global gpu

    def __init__(self, model, luna=True, segmentation=False):
        """
        :param model: model to be trained (sequential keras model)
        :param luna: boolean, True if dataset is Luna
        :param segmentation: boolean, True if the training includes segmentation
        """
        self.input_shape = [256, 256, 256]
        self.train_gen = None
        self.val_gen = None
        self.model = model
        self.segmentation = segmentation
        if luna:
            loader_model = LungsLoader()
            scans_ids = loader_model.get_scan_ids()
            print("Dataset size : ", len(scans_ids))
            if not segmentation:
                (width, height, depth) = self.input_shape
                train_ids, val_ids = train_test_split(scans_ids, test_size=0.2)
                self.train_gen = scan_generator(train_ids, width, height, depth, loop=True)
                self.val_gen = scan_generator(val_ids, width, height, depth, loop=True)
            else:
                (width, height, depth) = self.input_shape
                train_ids, val_ids = train_test_split(scans_ids, test_size=0.2)
                self.train_gen = scan_and_seg_generator(train_ids, width, height, depth, loop=True)
                self.val_gen = scan_and_seg_generator(val_ids, width, height, depth, loop=True)

    def train_model(self, gen_train=None, params_train=None, model_name="model_train"):
        """ :param : gen_train :  is a generator of the train set
            :param : params_train is a dictionary with the following keys : "optimizer": keras optimizer,
                        "loss_list": [list of keras losses,
                        "reg_param": float,
                        "seg_reg_param": float,
                        "initial_epoch": int,
                        "nb_epochs": int,
                        "steps_per_epoch": int
            :param : model_name is the name of the testing model

        """
        if params_train is None:
            params_train = {
                "optimizer": Adam(1e-4),
                "loss_list": [losses.Grad('l2').loss, losses.binary_dice],
                "reg_param": 0.5,
                "seg_reg_param": 1e-3,
                "initial_epoch": 0,
                "nb_epochs": 100,
                "steps_per_epoch": 100
            }
        if gen_train is not None:
            self.train_gen = gen_train
        if not self.segmentation:
            self.model.compile(
                        optimizer=params_train["optimizer"],
                        loss=params_train["loss_list"],
                        loss_weights=[1.0, params_train["reg_param"]]
                                                    )
        else:
            self.model.compile(
                        optimizer=params_train["optimizer"],
                        loss=params_train["loss_list"],
                        loss_weights=[1.0, params_train["reg_param"], params_train['seg_reg_param']]
                                                    )
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),model_name+"/")
        save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
        save_callback = ModelCheckpoint(save_file_name, verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=1e-3,
                                       patience=4,
                                       mode='auto')
        tbCallBack = TensorBoard(log_dir='./tbGraphs'+model_name, histogram_freq=0, write_graph=True, write_images=True)
        with tf.device(gpu):
            loss_h = self.model.fit_generator(
                                         self.train_gen,
                                         initial_epoch=params_train["initial_epoch"],
                                         epochs=params_train["nb_epochs"],
                                         callbacks=[save_callback, early_stopping, tbCallBack],
                                         steps_per_epoch=params_train["steps_per_epoch"],
                                         verbose=1,
                                         validation_data=self.val_gen,
                                         validation_steps=int(0.2 * params_train["steps_per_epoch"]))
        with open('history_'+model_name+'.json', 'w') as fp:
            json.dump(loss_h.history, fp)
        return None

    def test(self, val_gen=None, model_name='model'):
        """
        :param val_gen: generator for the validation set
        :param model_name: name of the model (filename)
        """
        if val_gen is not None:
            self.val_gen = val_gen
        pipeline_test_set(self.model, self.val_gen, model_name, self.segmentation)
        return None
