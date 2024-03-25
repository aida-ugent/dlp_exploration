from datetime import datetime

import pandas as pd


def load_highschool():
    events_url = (
        "http://www.sociopatterns.org/wp-content/uploads/2014/08/highschool_2012.csv.gz"
    )
    metadata_url = (
        "http://www.sociopatterns.org/wp-content/uploads/2015/09/metadata_2012.txt"
    )
    events = pd.read_csv(
        events_url,
        sep="\t",
        header=None,
        names=["t", "src", "dst", "Ci", "Cj"],
    ).drop(
        columns=["Ci", "Cj"]
    )  # Ci and Cj are redundant with node-level information
    # a = events["Ci"]
    # b = nodes.set_index("id").loc[events["src"].values]["class"]
    # np.all(a.values == b.values)
    # a = events["Cj"]
    # b = nodes.set_index("id").loc[events["dst"].values]["class"]
    # np.all(a.values == b.values)
    nodes = pd.read_csv(
        metadata_url,
        sep="\t",
        header=None,
        names=["raw_id", "class", "gender"],
    ).reset_index(names=["id"])
    events[["src", "dst"]]  # Parse timestamps
    events["datetime"] = events["t"].apply(datetime.fromtimestamp) + pd.Timedelta(
        hours=1
    )  # We add +1 because its winter time

    events["date"] = events["datetime"].apply(lambda x: x.date())
    events["src"] = nodes.set_index("raw_id").loc[events["src"]]["id"].values
    events["dst"] = nodes.set_index("raw_id").loc[events["dst"]]["id"].values
    return {
        "events": events,
        "nodes": nodes,
    }
