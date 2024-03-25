"""
Instantiate (or load) a link predictor and evaluate it on the one vs few task, which aims at discriminating the positive event from a short list of randomly sampled negative events.
"""

from tqdm import tqdm
import argparse
import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score


import numpy as np
import pandas as pd
from dynamic_link_prediction.dynamic_link_predictors import (
    EdgeBankLinkPredictor,
    PreferentialAttachmentLinkPredictor,
)
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

from common import df_to_temporal_data, train_val_test_split_dataframe

# Set random seed
from datasets import dataset_factories
from neg_sampling import edge_neg_sampler, dst_neg_sampler
from stats import (
    collect_edge_statistics,
    collect_node_statistics,
    label_edges,
    label_nodes,
)

link_predictor_factory = {
    "edgebank": EdgeBankLinkPredictor,
    "preferential_attachment": PreferentialAttachmentLinkPredictor,
}


def eval_one_vs_few(model, data_loader, neg_samplers):
    """
    One Vs few evaluation of the link predictor.
    """
    model.init()
    model.eval()

    out_dict = defaultdict(list)
    for batch in tqdm(data_loader):
        # calculate the score of the positive event
        pos_out = model(batch.src, batch.dst, batch.t, batch.msg).detach().cpu()
        out_dict["score_pos"] += pos_out.ravel().tolist()
        out_dict["src"] += batch.src.ravel().tolist()
        out_dict["dst"] += batch.dst.ravel().tolist()
        out_dict["t"] += batch.t.ravel().tolist()
        # calculate the score of the negative events
        for ns_name, neg_sampler in neg_samplers.items():
            src_neg, dst_neg = neg_sampler(
                batch.src.cpu().numpy(), batch.dst.cpu().numpy()
            )
            assert np.all(dst_neg >= min_dst)
            assert np.all(dst_neg <= max_dst)

            neg_out = model(
                torch.tensor(src_neg).long().to(model.device),
                torch.tensor(dst_neg).long().to(model.device),
                batch.t,
                batch.msg,
            ).cpu()  # Tensor of size (batch_size, 1)
            out_dict[f"score_{ns_name}"] += neg_out.ravel().tolist()
            # calculate the auc and ap scores
        # Memorize the positive event
        model.memorize(batch.src, batch.dst, batch.t, batch.msg)

    return out_dict


# pos_rank = (df["pos"].values[:, None] > df[cols[1:]].values).sum(axis=1) + 1
def cols_to_rank(df, score_cols):
    """
    For each column in score_cols, replace the raw scores
    by the rank of the score in the list of scores
    """
    for col in score_cols:
        col_others = [c for c in score_cols if c != col]
        rank_pessimistic = (df[col].values[:, None] < df[col_others].values).sum(
            axis=1
        ) + 1
        rank_optimistic = (df[col].values[:, None] <= df[col_others].values).sum(
            axis=1
        ) + 1

        df[col.replace("score", "rank")] = 0.5 * (rank_pessimistic + rank_optimistic)


from neg_sampling import dst_neg_sampler
from neg_sampling import edge_neg_sampler


def construct_neg_samplers(events):
    edge_statistics = collect_edge_statistics(events)
    node_statistics = collect_node_statistics(events)
    label_nodes(node_statistics, events)
    label_edges(edge_statistics, events)

    src_hist, dst_hist = edge_statistics.loc[
        edge_statistics["category"] == "historical", ["src", "dst"]
    ].values.T

    src_ind, dst_ind = edge_statistics.loc[
        edge_statistics["category"] == "inductive", ["src", "dst"]
    ].values.T

    src_ovl, dst_ovl = edge_statistics.loc[
        edge_statistics["category"] == "overlap", ["src", "dst"]
    ].values.T
    hist_sampler = lambda src, dst: edge_neg_sampler(src, dst, src_hist, dst_hist)

    ind_sampler = lambda src, dst: edge_neg_sampler(src, dst, src_ind, dst_ind)

    ovl_sampler = lambda src, dst: edge_neg_sampler(src, dst, src_ovl, dst_ovl)

    dst_nodes = np.unique(events.dst.values)
    dst_node_hist = (
        node_statistics.loc[dst_nodes]
        .loc[node_statistics["category"] == "historical"]
        .index.values
    )
    dst_node_ind = (
        node_statistics.loc[dst_nodes]
        .loc[node_statistics["category"] == "inductive"]
        .index.values
    )
    dst_node_ovl = (
        node_statistics.loc[dst_nodes]
        .loc[node_statistics["category"] == "overlap"]
        .index.values
    )

    hist_dst_sampler = lambda src, dst: dst_neg_sampler(src, dst, dst_node_hist)

    ind_dst_sampler = lambda src, dst: dst_neg_sampler(src, dst, dst_node_ind)

    ovl_dst_sampler = lambda src, dst: dst_neg_sampler(src, dst, dst_node_ovl)

    neg_samplers = {
        "historical_edge": hist_sampler,
        "inductive_edge": ind_sampler,
        "overlap_edge": ovl_sampler,
    }
    if len(dst_node_hist) > 0:
        neg_samplers["historical_dst"] = hist_dst_sampler
    else:
        print("Warning: no historical dst nodes found in the dataset")
    if len(dst_node_ind) > 0:
        neg_samplers["inductive_dst"] = ind_dst_sampler
    else:
        print("Warning: no inductive dst nodes found in the dataset")
    if len(dst_node_ovl) > 0:
        neg_samplers["overlap_dst"] = ovl_dst_sampler
    else:
        print("Warning: no overlap dst nodes found in the dataset")
    return neg_samplers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikipedia_tiny",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="edgebank",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/one_vs_few/",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
    )
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if args.use_cuda else "cpu"  # torch.device(args.device)
    torch.cuda.set_device(args.device)

    events, nodes = dataset_factories[args.dataset]().values()

    print(
        f"Running one vs few evaluation of {args.model} on {args.dataset} with {len(nodes)} nodes and {len(events)} events."
    )
    full_data = df_to_temporal_data(events).to(device)
    full_data_loader = TemporalDataLoader(
        full_data,
        batch_size=200,
    )
    _ = train_val_test_split_dataframe(events, val_ratio=0, test_ratio=0.15)
    splits = events.index

    #  Construct Negative sampler

    neg_samplers = construct_neg_samplers(events)
    min_dst = events.dst.min()
    max_dst = events.dst.max()

    # Model
    model = link_predictor_factory[args.model](
        temporal_data=full_data,
        device=device,
    )
    if args.saved_model_path is not None:
        artifact = torch.load(args.saved_model_path)
        model.load_state_dict(artifact["state_dict"])

    if args.saved_model_path is not None:
        fpath = os.path.basename(args.saved_model_path).replace(".pt", "_scores.csv")
    else:
        fpath = f"{args.model}_{args.dataset}_scores.csv"
    scores_path = os.path.join(
        args.output_dir,
        fpath,
    )
    scores_dict = eval_one_vs_few(model, full_data_loader, neg_samplers)
    scores_df = pd.DataFrame(scores_dict)
    scores_df["split"] = splits
    print("saving scores to", scores_path)
    scores_df.to_csv(scores_path)

    scores_col = [col for col in scores_df.columns if "score" in col]

    scores_cols_others = [
        col for col in scores_df.columns if "score" in col and col != "score_pos"
    ]

    # Save Test AUCs
    def calculate_auc(y_pos, y_neg):
        y_pred = np.concatenate([y_pos, y_neg])
        y_true = np.concatenate([np.ones_like(y_pos), np.zeros_like(y_neg)])
        return roc_auc_score(y_true, y_pred)

    test_mask = scores_df["split"] == "test"
    scores_df_test = scores_df.loc[test_mask]
    test_aucs = {}
    for col in scores_cols_others:
        y_pos = scores_df_test["score_pos"].values
        y_neg = scores_df_test[col].values
        auc = calculate_auc(y_pos, y_neg)
        test_aucs["auc_" + col] = auc

    auc_dir = os.path.join(args.output_dir, "aucs")
    os.makedirs(auc_dir, exist_ok=True)
    auc_path = os.path.join(
        auc_dir, os.path.basename(scores_path.replace("scores.csv", "aucs.csv"))
    )

    test_aucs = pd.Series(
        {
            "model": args.model,
            "dataset": args.dataset,
            "model_path": args.saved_model_path,
            **test_aucs,
        }
    )

    print("saving test aucs to", auc_path)
    test_aucs.to_csv(auc_path)
