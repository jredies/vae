import pathlib
import os
import importlib.util

import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def model_path():
    package_name = "vae"
    package_spec = importlib.util.find_spec(package_name)
    package_path = os.path.dirname(package_spec.origin)

    relative_folder_path = os.path.join(package_path, "stored_models")
    absolute_folder_path = os.path.abspath(relative_folder_path)

    return pathlib.Path(absolute_folder_path)
