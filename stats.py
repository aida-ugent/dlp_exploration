import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import minmaxscale
from pairing import SzudzikPair


def collect_node_statistics(df):
    """An in-place function that adds the following columns to the dataframe:
    - src_arrival_rank: the arrival rank of the source node
    - dst_arrival_rank: the arrival rank of the destination node
    - src_degree: the degree of the source node
    - dst_degree: the degree of the destination node
    Args:
        df (pd.DataFrame): a dataframe of events
    """

    df_dup = pd.concat([df.reset_index(), df.reset_index()]).sort_index()
    df_dup["node"] = np.transpose(np.vstack([df.src.values, df.dst.values])).ravel()
    df_dup["node_degree"] = df_dup.groupby("node")["t"].rank(method="first")
    df_dup["role"] = np.tile(["src", "dst"], len(df))
    df_node = (
        df_dup.groupby("node")
        .agg(
            t_min=("t", "min"),
            t_max=("t", "max"),
            node_count=("t", "count"),
            n_as_src=("role", lambda x: (x == "src").sum()),
            n_as_dst=("role", lambda x: (x == "dst").sum()),
        )
        .sort_values(["t_min", "t_max"], ascending=True)
    )

    df_node["rank"] = np.arange(len(df_node))

    df["src_arrival_rank"] = df.src.map(df_node["rank"])
    df["dst_arrival_rank"] = df.dst.map(df_node["rank"])

    df_dup = df_dup.set_index("role")

    df["src_degree"] = df_dup.loc["src", "node_degree"].values
    df["dst_degree"] = df_dup.loc["dst", "node_degree"].values

    return df_node


def collect_edge_statistics(df):
    df["edge_key"] = SzudzikPair.encode(df.src, df.dst)

    df["edge_degree"] = df.groupby("edge_key")["t"].rank(method="first").values

    df_edge = df.groupby("edge_key").agg(
        t_min=("t", "min"),
        t_max=("t", "max"),
        edge_count=("t", "count"),
        src=("src", "first"),
        dst=("dst", "first"),
    )
    df_edge.sort_values(["t_min", "t_max"], inplace=True)
    df_edge["rank"] = np.arange(len(df_edge))
    # df_edge.loc[df.edge_key, 'rank'].values
    df["edge_arrival_rank"] = df_edge.loc[df.edge_key, "rank"].values
    return df_edge


def label_nodes(node_stats, events):
    events = events.reset_index(drop=False).set_index("split")

    train_nodes = np.unique(events.loc["train", ["src", "dst"]].values)
    test_nodes = np.unique(events.loc["test", ["src", "dst"]].values)

    train_node_mask = node_stats.index.isin(train_nodes)
    test_node_mask = node_stats.index.isin(test_nodes)

    historical_node_mask = train_node_mask & ~test_node_mask
    overlap_node_mask = train_node_mask & test_node_mask
    inductive_node_mask = ~train_node_mask & test_node_mask

    node_stats.loc[historical_node_mask, "category"] = "historical"
    node_stats.loc[overlap_node_mask, "category"] = "overlap"
    node_stats.loc[inductive_node_mask, "category"] = "inductive"


def label_edges(edge_stats, events):
    train_edge_keys = np.unique(events.loc["train", "edge_key"])
    test_edge_keys = np.unique(events.loc["test", "edge_key"])

    edgekeys = edge_stats.index.values

    train_mask = np.isin(edgekeys, train_edge_keys)
    test_mask = np.isin(edgekeys, test_edge_keys)

    historical_mask = np.logical_and(train_mask, ~test_mask)
    overlap_mask = np.logical_and(train_mask, test_mask)
    inductive_mask = np.logical_and(test_mask, ~train_mask)

    edge_stats.loc[historical_mask, "category"] = "historical"
    edge_stats.loc[overlap_mask, "category"] = "overlap"
    edge_stats.loc[inductive_mask, "category"] = "inductive"


def temporal_node_traffic(src, dst, t):
    df = pd.DataFrame(
        {
            "src": src,
            "dst": dst,
            "absolute_t": t,
            "t": minmaxscale(t),
        }
    )
    df["node"] = df.apply(lambda x: [x["src"], x["dst"]], axis=1)
    df = df.explode("node")
    df["node"] = df["node"].astype(int)
    df_node = (
        df.groupby("node")
        .agg(
            t_min=("t", "min"),
            t_max=("t", "max"),
        )
        .sort_values(["t_min", "t_max"], ascending=True)
    )
    df_node["rank"] = np.arange(len(df_node))
    df_node["x_pos"] = minmaxscale(df_node["rank"].values)
    return df_node


def temporal_sender_traffic(src, t):
    df = pd.DataFrame(
        {
            "src": src,
            "absolute_t": t,
            "t": minmaxscale(t),
        }
    )

    df_sender = (
        df.groupby("src")
        .agg(t_min=("t", "min"), t_max=("t", "max"))
        .sort_values(["t_min", "t_max"], ascending=True)
    )
    df_sender["rank"] = np.arange(len(df_sender))
    df_sender["x_pos"] = minmaxscale(df_sender["rank"].values)
    return df_sender


def temporal_receiver_traffic(dst, t):
    df = pd.DataFrame(
        {
            "dst": dst,
            "absolute_t": t,
            "t": minmaxscale(t),
        }
    )

    df_receiver = (
        df.groupby("dst")
        .agg(t_min=("t", "min"), t_max=("t", "max"))
        .sort_values(["t_min", "t_max"], ascending=True)
    )
    df_receiver["rank"] = np.arange(len(df_receiver))
    df_receiver["x_pos"] = minmaxscale(df_receiver["rank"].values)
    return df_receiver


def temporal_edge_traffic(src, dst, t):
    df = pd.DataFrame(
        {
            "src": src,
            "dst": dst,
            "absolute_t": t,
            "t": minmaxscale(t),
        }
    )

    df["edge_key"] = SzudzikPair.encode(df["src"].values, df["dst"].values)

    df_edge = df.groupby(["edge_key"]).agg(
        t_min=("t", "min"),
        t_max=("t", "max"),
        edge_count=("t", "count"),
        src=("src", "first"),
        dst=("dst", "first"),
    )
    df_edge.sort_values(["t_min", "t_max"], inplace=True)
    df_edge["rank"] = np.arange(len(df_edge))
    df_edge["x_pos"] = minmaxscale(df_edge["rank"])
    return df_edge


def get_x_tet(src, dst, t):
    """
    A helper function that directly returns the x_tet values for each event
    """
    df = pd.DataFrame(
        {
            "src": src,
            "dst": dst,
            "absolute_t": t,
            "t": minmaxscale(t),
        }
    )

    df["edge_key"] = SzudzikPair.encode(df["src"].values, df["dst"].values)

    df_edge = df.groupby(["edge_key"]).agg(
        t_min=("t", "min"),
        t_max=("t", "max"),
        edge_count=("t", "count"),
        src=("src", "first"),
        dst=("dst", "first"),
    )
    df_edge.sort_values(["t_min", "t_max"], inplace=True)
    df_edge["rank"] = np.arange(len(df_edge))
    df_edge["x_pos"] = minmaxscale(df_edge["rank"])
    x_tet = df_edge.loc[df["edge_key"], "x_pos"].values
    return x_tet


def get_test_only_edges(events):
    """Calculate test only edges"""
    test_mask = events.index.get_level_values("split") == "test"
    train_mask = events.index.get_level_values("split") == "train"
    val_mask = events.index.get_level_values("split") == "val"
    edge_keys = SzudzikPair.encode(events.src, events.dst)
    keys_observed_during_test_only = (
        set(edge_keys.loc[test_mask])
        - set(edge_keys.loc[train_mask])
        - set(edge_keys.loc[val_mask])
    )
    src_test_only, dst_test_only = SzudzikPair.decode(
        np.array(list(keys_observed_during_test_only))
    )

    return src_test_only, dst_test_only


def get_test_only_nodes(events):
    """Calculate test only nodes"""

    node_cols = ["src", "dst"]
    nodes_observed_during_test_only = (
        set(events.loc["test", node_cols])
        - set(events.loc["train", node_cols])
        - set(events.loc["val", node_cols])
    )

    return np.array(list(nodes_observed_during_test_only))


def get_test_only_senders(events):
    """Calculate test only nodes"""

    src_observed_during_test_only = (
        set(events.loc["test", "src"])
        - set(events.loc["train", "src"])
        - set(events.loc["val", "src"])
    )

    return np.array(list(src_observed_during_test_only))


def get_test_only_destination(events):
    """Calculate test only nodes"""

    dst_observed_during_test_only = (
        set(events.loc["test", "dst"])
        - set(events.loc["train", "dst"])
        - set(events.loc["val", "dst"])
    )

    return np.array(list(dst_observed_during_test_only))


def get_temporal_edge_degree(src, dst, t):
    """
    For each event, return the number of past interactions of the edge just after the event.
    """
    df = pd.DataFrame({"src": src, "dst": dst, "t": t})

    edge_key = ["src", "dst"]

    return df.groupby(edge_key)["t"].rank(method="first").values


def get_temporal_node_degrees(src, dst, t):
    """
    For each event, return the number of past interactions of the source and the destination
    just after the event.
    """
    df = pd.DataFrame({"src": src, "dst": dst, "t": t})

    df_node = pd.concat([df.reset_index(), df.reset_index()]).sort_values("t")
    df_node["node"] = np.transpose(np.vstack([df.src.values, df.dst.values])).ravel()

    df_node["node_event_rank"] = df_node.groupby("node")["t"].rank(method="first")
    df_node["role"] = np.tile(["src", "dst"], len(df))
    df_node = df_node.set_index(["role"])

    src_degree = df_node.loc["src", "node_event_rank"].values
    dst_degree = df_node.loc["dst", "node_event_rank"].values

    return src_degree, dst_degree
