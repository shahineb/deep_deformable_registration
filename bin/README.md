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
├── train_ids.csv
├── val_ids.csv
└── test_ids.csv
```

It contains :

- `builder.pickle` : a serialized neural network builder instance which is used to build the model
- `config.pickle`: a serialized training configuration file containing all information about how training should be performed (losses, optimizers, epochs ...)
- `checkpoints`: directory where model weights checkpoints are stored
- `tensorboard`: directory where tensorboard logs are stored
- `train_ids.csv`, `val_ids.csv`, `test_ids.csv`: ids of luna scans used for training, validation and testing

(see more in wiki or in `tutorial.ipynb`)


## Run a training session

__Setup :__

Access `dashboard.ipynb` and go through the notebook to pick a session name, define model builder architecture, training configuration and training/validation/testing sets. Save all under session directory as specified above.

__Training :__

Run `python train.py --session_name=xxxxxx --gpu_id=x` where :
  - `--session_name`: name of the training session directory
  - `gpu_id`: optional, allows to switch gpu (default: `0`)


__Monitoring :__

Access `dashboard.ipynb` in monitoring section and specify a weight checkpoint file and evaluation metrics. Utilities are defined to visualize model's prediction over some samples and evaluate the model performances over testing set.
