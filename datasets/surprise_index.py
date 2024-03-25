import argparse
import json
import os
import sys

sys.path.append(os.getcwd())
import numpy as np

# if not os.path.exists(dataset_stats_path):
#     with open(dataset_stats_path, "w") as f:
#         json.dump({}, f)
import pandas as pd

from common import train_val_test_split_dataframe
from datasets import dataset_factories
from stats import collect_edge_statistics, collect_node_statistics

# print(sys.path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="uci",
    choices=[
        "uci",
        "highschool",
        "wikipedia",
        "wikipedia_tiny",
        "enron",
        "USLegis",
        "UNvote",
        "UNtrade",
        "SocialEvo",
        "mooc",
        "Flights",
        "reddit",
        "lastfm",
        "CanParl",
    ],
)

parser.add_argument("--split_ratio", type=float, default=0.15)

args = parser.parse_args()

events, nodes = dataset_factories[args.dataset]().values()

df_edge = collect_edge_statistics(events)

df_node = collect_node_statistics(events)


train_val_test_split_dataframe(events, 0, args.split_ratio)

df_edge["split"] = "overlap"

t_split = events.test_time
df_edge.loc[df_edge["t_max"] < t_split, "split"] = "historical"
df_edge.loc[df_edge["t_min"] > t_split, "split"] = "inductive"

num_inductive_edge = np.sum(df_edge["split"] == "inductive")
num_historical_edge = np.sum(df_edge["split"] == "historical")
num_overlap_edge = np.sum(df_edge["split"] == "overlap")


df_node["split"] = "overlap"
df_node.loc[df_node["t_max"] < t_split, "split"] = "historical"
df_node.loc[df_node["t_min"] > t_split, "split"] = "inductive"

num_inductive_nodes = np.sum(df_node["split"] == "inductive")
num_historical_nodes = np.sum(df_node["split"] == "historical")
num_overlap_nodes = np.sum(df_node["split"] == "overlap")
assert num_inductive_edge + num_historical_edge + num_overlap_edge == len(df_edge)
assert num_inductive_nodes + num_historical_nodes + num_overlap_nodes == len(df_node)

stats = pd.DataFrame(
    {
        "split_ratio": args.split_ratio,
        "Edge Surprise Index": num_inductive_edge
        / (num_inductive_edge + num_overlap_edge),
        "Node Surprise Index": num_inductive_nodes
        / (num_inductive_nodes + num_overlap_nodes),
    },
    index=[args.dataset],
)

dataset_stats_path = "output/datasets_stats.csv"

stats.to_csv(dataset_stats_path, mode="a", header=False)
