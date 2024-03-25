import os

import numpy as np
import pandas as pd

import common

from .highschool import load_highschool

DATA_PATH = os.path.expanduser("~/data/dgb")

DGB_Datasets = [
    "wikipedia",
    "lastfm",
    "reddit",
    "CanParl",
    "Contacts",
    "enron",
    "Flights",
    "mooc",
    "SocialEvo",
    "UNtrade",
    "UNvote",
    "USLegis",
]


# CanParl  Contacts  datasets_readme.md  enron  Flights  highschool_ct1  lastfm  mooc  reddit  SocialEvo  uci  UNtrade  UNvote  USLegis  wikipedia
def load_dgb(dataset="uci"):
    """
    These datasets from Poursafaei, F., Huang, S., Pelrine, K., & Rabbany, R. (2022). Dataset for "Towards Better Evaluation for Dynamic Link Prediction" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7213796 
    can be downloaded from  https://zenodo.org/record/7213796#.Y1cO6y8r30o

    Please change DATA_PATH with the path to the directory where you have downloaded the datasets.
    """
    fpath = os.path.join(DATA_PATH, dataset, f"ml_{dataset}.csv")
    events = pd.read_csv(fpath, index_col=0).rename(
        columns={
            "u": "src",
            "i": "dst",
            "ts": "t",
        }
    )
    fnode_path = os.path.join(DATA_PATH, dataset, f"ml_{dataset}_node.npy")
    nodes = np.load(fnode_path)

    return {"events": events, "nodes": nodes}


def load_wiki_tiny():
    events, nodes = load_dgb("wikipedia").values()
    events = events.iloc[:3000]
    return {"events": events, "nodes": nodes}



dataset_factories = {
    "uci": lambda: load_dgb("uci"),
    "highschool": load_highschool,
    "wikipedia": lambda: load_dgb("wikipedia"),
    "wikipedia_tiny": load_wiki_tiny,
    "enron": lambda: load_dgb("enron"),
    "USLegis": lambda: load_dgb("USLegis"),
    "UNvote": lambda: load_dgb("UNvote"),
    "UNtrade": lambda: load_dgb("UNtrade"),
    "SocialEvo": lambda: load_dgb("SocialEvo"),
    "mooc": lambda: load_dgb("mooc"),
    "Flights": lambda: load_dgb("Flights"),
    "reddit": lambda: load_dgb("reddit"),
    "lastfm": lambda: load_dgb("lastfm"),
    "CanParl": lambda: load_dgb("CanParl"),
}
