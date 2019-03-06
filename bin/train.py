import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, get_session


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(base_dir)

from src.networks.MariaNet import MariaNet
from src.networks.DiffeomorphicUnet import DiffeomorphicUnet
from src.training.config_file import ConfigFile
from src.training.luna_training import LunaTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True, type=Path,
                    help="name of training session to run")
parser.add_argument("--builder", required=True, type=str,
                    help="type of builder for neural net : {marianet, unet}")
parser.add_argument("--gpu_id", required=False, type=int,
                    help="GPU to set session on")
parser.add_argument("--weights", required=False, type=Path,
                    help="path to model weights to load before training")
parser.add_argument("--gen", required=False, type=str,
                    help="generator to use")


if __name__ == "__main__":
    args = parser.parse_args()
    session_dir = os.path.join(ConfigFile.bin_dir, args.session)

    # Load model
    if args.builder == "marianet":
        net_builder = MariaNet()
    elif args.builder == "unet":
        net_builder = DiffeomorphicUnet()
    else:
        raise UnboundLocalError("Unknown specified builder")

    net_builder.load(os.path.join(session_dir, ConfigFile.builder_filename))
    model = net_builder.build()

    # Setup device
    gpu_id = args.gpu_id or 0
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    set_session(tf.Session(config=tf_config))
    get_session().run(tf.global_variables_initializer())
    
    # Write weights path
    weights_path = args.weights
    if weights_path:
        weights_path = os.path.join(session_dir, ConfigFile.checkpoints_dirname, weights_path)

    # Load training and validation sets
    train_ids = pd.read_csv(os.path.join(session_dir, LunaTrainer.train_ids_filename)).values.squeeze()
    val_ids = pd.read_csv(os.path.join(session_dir, LunaTrainer.val_ids_filename)).values.squeeze()
    
    # Setup trainer
    trainer = LunaTrainer(model=model,
                          device=gpu,
                          config_path=os.path.join(session_dir, ConfigFile.pickle_filename),
                          weights_path=weights_path,
                          tensorboard=True)

    # Train
    trainer.fit(train_ids, val_ids, generator=args.gen, use_affine=False)
