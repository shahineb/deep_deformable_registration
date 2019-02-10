import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

import sys
sys.path.append(base_dir)

from src.networks.MariaNet import MariaNet
from src.training.config_file import ConfigFile
from src.training.luna_training import LunaTrainer
