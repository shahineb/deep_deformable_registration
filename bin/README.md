# Training sessions

A training session is intialized as a directory containing all the needed information to reproduce the so told training.

```
session_name
├── builder.pickle
├── config.pickle
├── checkpoints
│   └── chkpt_xx.h5
├── tensorboard
│   └── tensorboard logs
├── observations
│   └── observations_xx
├── scores
├── train_ids.csv
├── val_ids.csv
├── test_ids.csv
└── logs.txt
```

It contains :

- `builder.pickle` : a serialized neural network builder instance which is used to build the model
- `config.pickle`: a serialized training configuration file containing all information about how training should be performed (losses, optimizers, epochs ...)
- `checkpoints`: directory where model weights checkpoints are stored
- `tensorboard`: directory where tensorboard logs are stored
- `observations`: directory where model output is saved in png format for each epoch
- `scores`: directory where model evaluation dataframes are stored as csv files
- `train_ids.csv`, `val_ids.csv`, `test_ids.csv`: ids of luna scans used for training, validation and testing
- `logs.txt`: output training logs

(see more in `tutorial.ipynb`) --> deprecated tutorial, to update


## Run a training session

__Setup :__

Access `dashboard.ipynb` and go through the notebook to pick a session name, define model builder architecture, training configuration and training/validation/testing sets. Save all under session directory as specified above.

__Training :__

Run `python train.py --session --builder --gen --gpu_id --weights > session_name/logs.txt` where :
  - `--session`: name of the training session directory
  - `--builder`: network to use in {`marianet`, `unet`}
  - `--gen`: generator to use in {`luna`, `luna_seg`, `atlas`, `atlas_seg`}
  - `--gpu_id`: optional, allows to switch gpu (default: `0`)
  - `--weights`: optional, checkpoints weights to load for training formatted as `chkpt_xx.h5`

__Monitoring :__

Access `dashboard.ipynb` in monitoring section and specify a weight checkpoint file and evaluation metrics. Utilities are defined to visualize model's prediction over some samples and evaluate the model performances over testing set.
