# Enrich the sns.scatterplot with the run information

import argparse
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import TemporalData

DGB_DATA_PATH = os.path.expanduser("~/data/dgb")


def to_tensor(*args):
    return (torch.tensor(arg) for arg in args)


def to_numpy(*args):
    return (arg.cpu().detach().numpy() for arg in args)


class EarlyStopping:
    def __init__(self, min_delta=0.0, patience=10):
        self.min_delta = min_delta
        self.patience = patience
        self.count = 0
        self.best_value = None

    def __call__(self, value):
        """
        check if the value increases from the best value by at least min_delta
        """
        if self.best_value is None:
            self.best_value = value
            return False
        if value - self.best_value > self.min_delta:
            self.best_value = value
            self.count = 0
        else:
            self.count += 1
        return self.count > self.patience


def add_common_arguments(parser):
    parser.add_argument("--dataset", type=str, default="wikipedia")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikipedia")

    args = parser.parse_args()
    return args


def save_results(results, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(results, f)


def get_last_runs(run_dir, n_last=10):
    """Get the last n_last runs."""

    run_collection = RunCollection(run_dir, n_last=n_last)
    return run_collection.df.dropna()


def load_json(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r") as f:
        config = json.load(f)
    return config


def minmaxscale(x):
    return (x - min(x)) / (max(x) - min(x))


def set_theme(fontsize=25):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "font.family": "serif",
            # "annotation.fontsize": fontsize,
        }
    )
    sns.set_style(
        "whitegrid",
        rc={
            "axes.edgecolor": "0.15",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "font.family": "serif",
            # "axes.titlesize": fontsize,
            # "axes.labelsize": fontsize,
            # "xtick.labelsize": fontsize,
            # "ytick.labelsize": fontsize,
            # "legend.fontsize": fontsize,
            # "legend.title_fontsize": fontsize,
        },
    )


class Run:
    def __init__(self, path) -> None:
        self.config = load_json(path + "/config.json")
        self.config["id"] = path.split("/")[-1]
        self.path = path
        self.metrics = load_json(path + "/metrics.json")
        self.df = pd.Series(self.config)
        self.checkpoints = [path for path in os.listdir(path) if path.endswith(".pt")]
        if len(self.checkpoints) == 0:
            print(f"Warning: no checkpoint found in {path}")


# List dir absoulte path


class RunCollection:
    def __init__(self, dir, n_last=None) -> None:
        self.runs = [Run(os.path.join(dir, path)) for path in os.listdir(dir)[-n_last:]]
        self.dir = dir
        self.df = pd.DataFrame([run.config for run in self.runs])


def torch_compat(func):
    def wrapper(*torch_args):
        args = []
        there_is_tensor = False
        for arg in torch_args:
            if torch.is_tensor(arg):
                # If the input is a PyTorch tensor, convert it to a NumPy array
                device = arg.device
                there_is_tensor = True
                arg = arg.cpu().numpy()
            args.append(arg)
        out = func(*args)  # eventually this is a tuple of output
        if there_is_tensor:
            # Convert the output back to a PyTorch tensor
            out = torch.tensor(out).to(device)

        return out

    return wrapper


def numpy_compat(func):
    def wrapper(*numpy_args):
        args = []
        there_is_numpy = False
        for arg in numpy_args:
            if isinstance(arg, np.ndarray):
                # If the input is a PyTorch tensor, convert it to a NumPy array
                there_is_numpy = True
                arg = torch.from_numpy(arg)
            args.append(arg)
        out = func(*args)
        if there_is_numpy:
            out = out.numpy()

        return out

    return wrapper


def traffic_plot_theme(fontsize=25, rc={}):
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "font.family": "serif",
            **rc,
        },
    )


def traffic_plot_theme(fontsize=25, rc={}):
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "font.family": "serif",
            **rc,
        },
    )


def train_val_test_split_dataframe(df, val_ratio, test_ratio):
    """
    In place function to split a dataframe into train, val and test sets.
    A 'split' column gets added to the dataframe.
    """
    val_time, test_time = list(
        np.quantile(df["t"], [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )
    df["t_scaled"] = minmaxscale(df["t"])
    val_time_scaled, test_time_scaled = list(
        np.quantile(df["t_scaled"], [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )

    train_mask = (df["t"] < val_time).values
    val_mask = ((df["t"] >= val_time) & (df["t"] < test_time)).values
    test_mask = (df["t"] >= test_time).values

    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    df.set_index("split", inplace=True)
    df.test_time = test_time
    df.val_time = val_time
    df.val_time_scaled = val_time_scaled
    df.test_time_scaled = test_time_scaled
    df.val_ratio = val_ratio
    df.test_ratio = test_ratio
    # return df
    train_mask = df.index == "train"
    val_mask = df.index == "val"
    test_mask = df.index == "test"
    return train_mask, val_mask, test_mask


def df_to_temporal_data(df):
    return TemporalData(
        src=torch.tensor(df["src"].values),
        dst=torch.tensor(df["dst"].values),
        t=torch.tensor(df["t"].values).long(),
        msg=torch.tensor(df["label"].values).reshape(-1, 1).float(),
    )


def set_random_seed():
    import random

    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r") as f:
        config = json.load(f)
    return config


def minmaxscale(x):
    return (x - min(x)) / (max(x) - min(x))


def set_theme(fontsize=25):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({"font.size": fontsize})
    sns.set_style(
        "whitegrid",
        rc={
            "axes.edgecolor": "0.15",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "font.family": "serif",
        },
    )


class Run:
    def __init__(self, path) -> None:
        self.config = load_json(path + "/config.json")
        self.config["id"] = path.split("/")[-1]
        self.path = path
        self.metrics = load_json(path + "/metrics.json")
        self.df = pd.Series(self.config)


# List dir absoulte path


class RunCollection:
    def __init__(self, dir, n_last=None) -> None:
        self.runs = [Run(os.path.join(dir, path)) for path in os.listdir(dir)[-n_last:]]
        self.dir = dir
        self.df = pd.DataFrame([run.config for run in self.runs])


def torch_compat(func):
    def wrapper(*torch_args):
        args = []
        there_is_tensor = False
        for arg in torch_args:
            if torch.is_tensor(arg):
                # If the input is a PyTorch tensor, convert it to a NumPy array
                device = arg.device
                there_is_tensor = True
                arg = arg.cpu().numpy()
            args.append(arg)
        out = func(*args)  # eventually this is a tuple of output
        if there_is_tensor:
            # Convert the output back to a PyTorch tensor
            out = torch.tensor(out).to(device)

        return out

    return wrapper


def numpy_compat(func):
    def wrapper(*numpy_args):
        args = []
        there_is_numpy = False
        for arg in numpy_args:
            if isinstance(arg, np.ndarray):
                # If the input is a PyTorch tensor, convert it to a NumPy array
                there_is_numpy = True
                arg = torch.from_numpy(arg)
            args.append(arg)
        out = func(*args)
        if there_is_numpy:
            out = out.numpy()

        return out

    return wrapper


def traffic_plot_theme(fontsize=25, rc={}):
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "font.family": "serif",
            **rc,
        },
    )
