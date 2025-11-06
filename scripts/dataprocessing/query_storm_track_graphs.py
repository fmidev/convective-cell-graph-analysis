"""Query and store storm tracks from the database."""

import argparse
import logging
import os
from copy import copy
from datetime import datetime
from pathlib import Path
import time
from functools import partial
import gzip
from collections import defaultdict
from itertools import product
from functools import wraps

import dask.bag as db
import geopandas as gpd
import pandas as pd
import polars as pl
from tqdm import tqdm
from sqlalchemy.orm import Session
import networkx as nx
import numpy as np

from database import get_engine
from utils.config_utils import load_config


SQL_QUERY = """
WITH RECURSIVE sub_tree AS (
  SELECT
    start_cell.timestamp as start_time,
    start_cell.identifier as start_id,
    start_cell.timestamp,
    start_cell.identifier,
    start_cell.method,
    start_cell.prev_identifiers,
    start_cell.next_identifiers,
    1 as depth

  FROM {CELLS_WITH_PARENTS_CHILDREN_VIEW} as start_cell
  WHERE 1=1
    AND start_cell.method = '{method}'
    AND start_cell.timestamp  >= '{start_time}'
    AND start_cell.timestamp  < '{end_time}'
    AND TRUE = ANY(SELECT unnest(start_cell.prev_identifiers) IS NULL)

  UNION

  SELECT
    st.start_time,
    st.start_id,
    next_cell.timestamp,
    next_cell.identifier,
    next_cell.method,
    next_cell.prev_identifiers,
    next_cell.next_identifiers,
    st.depth + 1 as depth
  FROM {CELLS_WITH_PARENTS_CHILDREN_VIEW} as next_cell, sub_tree st

  WHERE 1=1
    AND st.identifier = ANY(next_cell.prev_identifiers)
    AND next_cell.timestamp = st.timestamp + interval '5 minutes'
    AND next_cell.method = st.method
)
SELECT * FROM sub_tree
"""


def query_storm_tracks(config, startdate, enddate, method):
    """
    Query storm tracks from the database.

    Parameters
    ----------
    config : dict
        Configuration.
    start_date : datetime
        Start of interval.
    end_date : datetime
        End of interval.
    method : str
        Cell identification method to be queried.

    """
    print(f"Querying storm tracks for {startdate} - {enddate} method {method}.")
    outpath = Path(config.output["storage_path"].format(method=method))
    outpath.mkdir(parents=True, exist_ok=True)
    # subpath = outpath / f"{method}_{startdate.strftime('%Y%m%d')}"

    engine = get_engine(config.cell_database)

    start_date = startdate
    end_date = enddate  # + pd.Timedelta(days=1)

    sql_query = SQL_QUERY.format(
        start_time=start_date.strftime("%Y-%m-%d %H:%M"),
        end_time=end_date.strftime("%Y-%m-%d %H:%M"),
        method=method,
        CELLS_WITH_PARENTS_CHILDREN_VIEW=config.cell_database.table,
    )

    with Session(engine) as session:
        # time the query
        start_time = time.time()
        # track_df = pd.read_sql(sql=sql_query, con=session.bind)
        track_df = pl.read_database(query=sql_query, connection=session.bind)

        if track_df.is_empty():
            logging.warning(f"No data found for method {method} between {start_date} and {end_date}.")
            return

        if track_df.height < 3:
            logging.warning(
                f"Not enough data found for method {method} between {start_date} and {end_date} (only {track_df.height} cells)."
            )
            return

        # Convert to pandas dataframe
        end_time = time.time()
        tqdm.write(f"Query took {end_time - start_time:.2f} seconds.")

    track_df = track_df.sort(["start_time", "start_id", "method", "timestamp", "identifier"])
    df_ = track_df.to_pandas()

    graph = nx.DiGraph()

    def generate_node_id(timestamp, identifier):
        return f"{timestamp:%Y-%m-%dT%H:%M:%S}_{int(identifier)}"

    def dateformat(x):
        return x.strftime("%Y-%m-%dT%H:%M:%S")

    for _, row in df_.iterrows():
        data_dict = row.to_dict()
        data_dict["timestamp"] = dateformat(row["timestamp"])  # strftime("%Y-%m-%d %H:%M")
        data_dict["start_time"] = dateformat(row["start_time"])  # strftime("%Y-%m-%d %H:%M")
        data_dict["prev_identifiers"] = row["prev_identifiers"].tolist()
        data_dict["next_identifiers"] = row["next_identifiers"].tolist()

        # Add the current cell as a node in the graph
        graph.add_node(generate_node_id(row["timestamp"], row["identifier"]), **data_dict)

    for _, row in df_.iterrows():
        # Add edges connecting the current cell to its previous cells
        try:
            if len(row["prev_identifiers"]) > 0 and np.isfinite(row["prev_identifiers"][0]):
                prev_timestamp = row["timestamp"] - pd.Timedelta(minutes=5)
                for prev_id in row["prev_identifiers"]:

                    # Check that the node exists before adding the edge
                    if graph.has_node(generate_node_id(prev_timestamp, prev_id)):
                        graph.add_edge(
                            generate_node_id(prev_timestamp, prev_id),
                            generate_node_id(row["timestamp"], row["identifier"]),
                        )
        except TypeError:
            logging.debug(f"Error in row {row} for day {startdate} - {enddate} and method {method}.")
            return

        # Add edges connecting the current cell to its next cells
        if len(row["next_identifiers"]) > 0 and np.isfinite(row["next_identifiers"][0]):
            next_timestamp = row["timestamp"] + pd.Timedelta(minutes=5)
            for next_id in row["next_identifiers"]:
                # Check that the node exists before adding the edge
                if graph.has_node(generate_node_id(next_timestamp, next_id)):
                    graph.add_edge(
                        generate_node_id(row["timestamp"], row["identifier"]),
                        generate_node_id(next_timestamp, next_id),
                    )

    subgraphs = [graph.subgraph(g).copy() for g in nx.connected_components(graph.to_undirected())]

    graph_id_counters = defaultdict(int)
    # For each subgrap, create a graph id and store in a file

    cell_list = []

    for i, subgraph in enumerate(subgraphs):

        min_date_in_subgraph = min([datetime.fromisoformat(node[1]["timestamp"]) for node in subgraph.nodes(data=True)])
        max_date_in_subgraph = max([datetime.fromisoformat(node[1]["timestamp"]) for node in subgraph.nodes(data=True)])

        subpath = outpath / f"{method}_{min_date_in_subgraph.strftime('%Y%m%d')}"
        subpath.mkdir(parents=True, exist_ok=True)

        # Create a new graph id
        graph_id_datestr = (
            f"{min_date_in_subgraph.strftime('%Y%m%d%H%M')}_{max_date_in_subgraph.strftime('%Y%m%d%H%M')}"
        )
        graph_id_num = graph_id_counters[graph_id_datestr]
        graph_id = f"{graph_id_datestr}_{graph_id_num}"
        graph_id_counters[graph_id_datestr] += 1
        for key, node in subgraph.nodes(data=True):
            # Add the graph id to the node attributes
            subgraph.nodes[key]["graph_id"] = graph_id
            cell_list.append(
                (
                    node["timestamp"],
                    node["identifier"],
                    method,
                    min_date_in_subgraph,
                    max_date_in_subgraph,
                    graph_id_num,
                    subgraph.number_of_nodes(),
                    subgraph.number_of_edges(),
                )
            )

        filename = f"track_graph_{method}_{graph_id}.gml"
        nx.write_gml(subgraph, subpath / filename, stringizer=stringizer)

    df = pd.DataFrame(
        cell_list,
        columns=[
            "timestamp",
            "identifier",
            "method",
            "min_timestamp",
            "max_timestamp",
            "graph_identifier",
            "num_nodes",
            "num_edges",
        ],
    )
    # Save the cell list to a database
    with Session(engine) as session:
        df.to_sql(
            "track_graphs",
            schema="raincells",
            con=session.bind,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )


def stringizer(rep):
    """
    Stringizer for the node attributes.

    Parameters
    ----------
    node : tuple
        Node to be stringized.

    Returns
    -------
    str
        Stringized node.
    """
    if isinstance(rep, tuple):
        return f"{rep[0]}_{rep[1]}"
    elif isinstance(rep, datetime):
        return rep.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        return str(rep)


def destringizer(rep):
    """
    Destringizer for the node attributes.

    Parameters
    ----------
    node : str
        Node to be destringized.

    Returns
    -------
    tuple
        Destringized node.
    """
    from ast import literal_eval

    if "_" in rep:
        return tuple(rep.split("_"))
    elif "T" in rep:
        return datetime.strptime(rep, "%Y-%m-%dT%H:%M:%S")
    else:
        if isinstance(rep, str):
            orig_rep = rep
            try:
                return literal_eval(rep)
            except SyntaxError as err:
                raise ValueError(f"{orig_rep!r} is not a valid Python literal") from err
        else:
            raise ValueError(f"{rep!r} is not a string")


def star(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # print(args[0])
        return f(*args[0][0], args[0][1], *args[1:], **kwargs)

    return wrapper


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("configpath", type=str, help="Config file path with database configuration.")
    argparser.add_argument(
        "datelist", type=str, help="Path to file containing list of dates to query, as startdate,enddate"
    )
    # argparser.add_argument("start_date", type=str, help="Start date of the interval, as YYYYmmdd.")
    # argparser.add_argument("end_date", type=str, help="End date of the interval, as YYYYmmdd.")
    argparser.add_argument("-n", "--nworkers", type=int, default=1, help="Number of workers.")
    # argparser.add_argument("--methods", type=str, nargs="+", help="Cell identification methods to be queried.")
    args = argparser.parse_args()

    confpath = Path(args.configpath)
    conf = load_config(confpath)

    methods = conf.settings.cell_identification_methods

    # start_date = datetime.strptime(args.start_date, "%Y%m%d")
    # end_date = datetime.strptime(args.end_date, "%Y%m%d")
    # days = pd.date_range(start_date, end_date, freq="1d").to_list()
    # days = [day for day in days if day.month in [5, 6, 7, 8, 9]]

    dates = pd.read_csv(args.datelist, header=None, names=["start_date", "end_date"])
    dates["start_date"] = pd.to_datetime(dates["start_date"], format="%Y%m%d%H%M")
    dates["end_date"] = pd.to_datetime(dates["end_date"], format="%Y%m%d%H%M")

    # import ipdb

    # ipdb.set_trace()

    func = partial(query_storm_tracks, conf)
    if args.nworkers > 1:
        from dask.distributed import Client

        # register progress bar
        from dask.diagnostics import ProgressBar

        ProgressBar().register()

        scheduler = "processes" if args.nworkers > 1 else "single-threaded"
        client = Client(n_workers=args.nworkers, threads_per_worker=1)

        bag = db.from_sequence(product(zip(dates["start_date"].tolist(), dates["end_date"].to_list()), methods))
        bag.map(star(func)).compute()
    else:
        for (start, end), method in tqdm(
            product(zip(dates["start_date"].tolist(), dates["end_date"].to_list()), methods)
        ):
            # tqdm.write(f"Querying storm tracks for day {day} method {method}.")
            query_storm_tracks(conf, start, end, method)
