import json
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)
