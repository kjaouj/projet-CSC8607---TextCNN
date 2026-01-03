"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import os
import json
import torch
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config):
    device_name = config["train"].get("device", "auto")
    if device_name == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_name
    return torch.device(device_str)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=4)

import yaml

def load_config(path):
    """
    Loads a YAML configuration file and returns it as a Python dict.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
