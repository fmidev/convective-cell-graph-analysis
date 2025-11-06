import argparse
import logging
import logging.config
import os
import shutil
import sys
import copy
from datetime import datetime
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import yaml
from dask.diagnostics import ProgressBar
from sqlalchemy.orm import Session

from database import get_engine
from utils import graph_utils
from utils.config_utils import load_config

GRAPH_SQL_QUERY = """
    SELECT DISTINCT method, min_timestamp, max_timestamp, graph_identifier, num_nodes, num_edges
	FROM raincells.track_graphs

    WHERE 1=1
    AND min_timestamp >= '{startdate}'
    AND max_timestamp < '{enddate}'
    AND extract(epoch from max_timestamp - min_timestamp) / 60 > {min_duration}

    ORDER BY min_timestamp
"""

# Setup logging
with open("logconf.yaml", "rt") as f:
    log_config = yaml.safe_load(f.read())
    f.close()
logging.config.dictConfig(log_config)
logging.captureWarnings(True)
logger = logging.getLogger(Path(__file__).stem)


def get_split_merge_trajectory_data_for_graph(row, config: dict, prune_t0=True, prune_duplicates=True) -> pl.DataFrame:
    """Get trajectory data for a given trajectory ID."""
    typename = "split_merge"
    type_ext = ""
    if prune_t0:
        typename += "_prunet0"
        type_ext += "_prunet0"
    if prune_duplicates:
        typename += "_prunedup"
        type_ext += "_prunedup"

    # Read track graph from file
    method = row["method"]
    min_timestamp = row["min_timestamp"].strftime("%Y%m%d%H%M")
    max_timestamp = row["max_timestamp"].strftime("%Y%m%d%H%M")
    identifier = row["graph_identifier"]

    graph_path = Path(row["min_timestamp"].strftime(config["track_graphs"]["storagepath"]).format(method=method))
    graph_filename = config["track_graphs"]["graph_filename_template"].format(
        method=method,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        identifier=identifier,
    )

    graph_file = graph_path / graph_filename
    graph_name = Path(graph_file).stem

    num_steps_to_hop = config["track_graphs"]["num_steps_to_hop"]

    # print(f"Processing graph {idx}: {graph_file} ({min_timestamp} - {max_timestamp})")
    if not graph_file.exists():
        logger.warning(f"Graph file {graph_file} does not exist.")
        return

    if not config["output"]["overwrite_existing"]:
        # If the output files already exist, skip processing
        outpath = Path(config["output"]["storagepath"]) / config["output"]["intermediate_subfolder"]
        filename_all = config["output"]["all_trajectories_intermediate_filename"].format(
            graph_name=graph_name,
            type=typename,
        )

        if (outpath / filename_all).exists():
            logger.info(f"Skipping processing for graph {graph_name} as output files already exist.")
            return

    G = nx.read_gml(graph_file, destringizer=graph_utils.string_decoder)
    # Find nodes that are merged from two or more nodes
    merged_nodes = set([n for n, deg in G.in_degree if deg > 1])
    split_nodes = set([n for n, deg in G.out_degree if deg > 1])
    split_merge_nodes = set([n for n in G.nodes if G.in_degree(n) > 1 and G.out_degree(n) > 1])

    merged_nodes = merged_nodes - split_merge_nodes
    split_nodes = split_nodes - split_merge_nodes

    preds_succs = {}
    nodetypes = {
        f"merged{type_ext}": merged_nodes,
        f"split{type_ext}": split_nodes,
        f"split-merge{type_ext}": split_merge_nodes,
    }
    nodetypes_selected = {k: v.copy() for k, v in nodetypes.items()}

    if prune_t0:
        # Remove nodes that have predecessors of the same type within num_steps_to_hop
        for ttype, nodes in nodetypes.items():
            for n in nodes:

                all_predecessors = graph_utils.get_all_predecessors(n, G, max_level=num_steps_to_hop)

                if any(pred in nodetypes[ttype] for pred in all_predecessors):
                    nodetypes_selected[ttype].remove(n)
                    continue

    for ttype, nodes in nodetypes_selected.items():
        for n in nodes:
            predecessors = graph_utils.get_predecessors(n, G, max_level=num_steps_to_hop)
            successors = graph_utils.get_successors(n, G, max_level=num_steps_to_hop)

            if "event" not in G.nodes[n]:
                G.nodes[n]["event"] = f"{ttype}_midnode."
            else:
                G.nodes[n]["event"] += f"{ttype}_midnode."

            preds_succs[n] = {
                "predecessors": predecessors,
                "successors": successors,
                "type": ttype,
            }

    keep_midnodes_all = []
    if prune_duplicates:
        for event in nodetypes_selected.keys():
            preds_succs_event = {k: v for k, v in preds_succs.items() if v["type"] == event}
            # Prune overlapping trajectories
            keep_midnodes_num_nodes = graph_utils.prune_overlapping_trajectories(
                preds_succs=preds_succs_event,
                graph=G,
            )
            keep_midnodes_all.extend(keep_midnodes_num_nodes)
    else:
        keep_midnodes_all = list(preds_succs.keys())

    #     graph_utils.plot_graph(G, "test_graph", time_resolution="5min", outpath=Path("."), ext=[ "pdf"])
    #     import ipdb

    #     ipdb.set_trace()
    # Get time series data for the merged and split nodes
    tseries = graph_utils.get_timeseries(G, preds_succs, engine=engine, **config["subgraphs"])

    if tseries is None:
        logger.warning(f"No time series data for graph {graph_name}.")
        return

    logger.info(f"Processing graph {graph_name} with {tseries.height} time series entries.")

    tseries = tseries.with_columns(
        pl.col("timestamp").count().over("type", "t0_node", "level").alias("num_cells_at_level")
    )
    tseries = tseries.filter((pl.col("t0_node").is_in(keep_midnodes_all)))

    if config["trajectory_selection"]["drop_database_values"]:
        drop_columns = [c for c in tseries.columns if "." in c]
        tseries = tseries.drop(drop_columns)
        # filtered_tseries = filtered_tseries.drop(drop_columns)

    # Write results to parquet files
    outpath = Path(config["output"]["storagepath"]) / config["output"]["intermediate_subfolder"]
    outpath.mkdir(parents=True, exist_ok=True)
    filename_all = config["output"]["all_trajectories_intermediate_filename"].format(
        graph_name=graph_name,
        type=typename,
    )
    tseries.write_parquet(outpath / filename_all)
    logger.info(f"Saved {tseries.height} rows to {outpath / filename_all}")

    return


def get_cell_condition_trajectory_data_for_graph(
    row, config: dict, prune_t0=True, prune_duplicates=True
) -> pl.DataFrame:
    """Get trajectory data for a given trajectory ID."""
    typename = "cell_conditions"
    type_ext = ""
    if prune_t0:
        typename += "_prunet0"
        type_ext += "_prunet0"
    if prune_duplicates:
        typename += "_prunedup"
        type_ext += "_prunedup"

    # Read track graph from file
    method = row["method"]
    min_timestamp = row["min_timestamp"].strftime("%Y%m%d%H%M")
    max_timestamp = row["max_timestamp"].strftime("%Y%m%d%H%M")
    identifier = row["graph_identifier"]

    graph_path = Path(row["min_timestamp"].strftime(config["track_graphs"]["storagepath"]).format(method=method))
    graph_filename = config["track_graphs"]["graph_filename_template"].format(
        method=method,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        identifier=identifier,
    )

    graph_file = graph_path / graph_filename
    graph_name = Path(graph_file).stem

    num_steps_to_hop = config["track_graphs"]["num_steps_to_hop"]
    pred_dist_limit = config["trajectory_selection"].get("predecessors_distance_limit", 140)

    if not graph_file.exists():
        logger.warning(f"Graph file {graph_file} does not exist.")
        return

    if not config["output"]["overwrite_existing"]:
        # If the output files already exist, skip processing
        outpath = Path(config["output"]["storagepath"]) / config["output"]["intermediate_subfolder"]
        filename_all = config["output"]["all_trajectories_intermediate_filename"].format(
            graph_name=graph_name,
            type=typename,
        )
        if (outpath / filename_all).exists():
            logger.info(f"Skipping processing for graph {graph_name} as output files already exist.")
            return

    G = nx.read_gml(graph_file, destringizer=graph_utils.string_decoder)

    condition_conf = config["trajectory_selection"]["select_cell_conditions"]

    control_conditions = [k for k in condition_conf.keys() if "control" in k]
    original_conditions = [k for k in condition_conf.keys() if "control" not in k]

    qty_stats = [q for k in original_conditions for q in list(condition_conf[k].keys())]
    qty_stats += [q for k in control_conditions for _, v in condition_conf[k].items() for q in list(v.keys())]
    qty_stats = set([q for q in qty_stats if "." in q])  # Remove duplicates
    quantities = set([q.split(".")[0] for q in qty_stats])
    statistics = set([q.split(".")[1] for q in qty_stats])

    cell_data = graph_utils.get_cell_data(
        G,
        quantities=quantities,
        statistics=statistics,
        engine=engine,
    )
    if cell_data.height == 0:
        logger.warning(f"No cell data for graph {graph_name}.")
        return
    cell_data = cell_data.with_columns(
        label=pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S")
        + pl.col("identifier").map_elements(lambda x: f"_{x}", return_dtype=pl.String)
    )

    all_tseries = []
    # all_filtered_tseries = []
    keep_midnodes_all = []

    # If the group is control, we want to combine all groups into one group after finding the candidate nodes
    # i.e. we want to find the candidate nodes based on multiple conditions
    # (e.g. rate limits change as function of area)
    for name, conditions in condition_conf.items():

        if "control" in name:

            cell_candidates = []
            for subname, subconditions in conditions.items():

                qtys = [q for q in subconditions.keys() if "." in q]
                available_qtys = set(cell_data.columns)
                all_available = np.all([q in available_qtys for q in qtys])

                if not all_available:
                    logger.warning(
                        f"Not all quantities are available in cell data for graph {graph_name} with conditions {name}."
                    )
                    continue

                # # Combine all conditions
                t0_cell_conditions = graph_utils.build_filter_from_conditions(subconditions)

                # Filter cell data based on the conditions
                cell_candidates.append(cell_data.filter(pl.all_horizontal(t0_cell_conditions)).sort(by="timestamp"))

            cell_candidates = pl.concat(cell_candidates, how="diagonal_relaxed")
        else:
            qtys = [q for q in conditions.keys() if "." in q]
            available_qtys = set(cell_data.columns)
            all_available = np.all([q in available_qtys for q in qtys])
            if not all_available:
                logger.warning(
                    f"Not all quantities are available in cell data for graph {graph_name} with conditions {name}."
                )
                continue

            # # Combine all conditions
            t0_cell_conditions = graph_utils.build_filter_from_conditions(conditions)

            # Filter cell data based on the conditions
            cell_candidates = cell_data.filter(pl.all_horizontal(t0_cell_conditions)).sort(by="timestamp")

        if cell_candidates.height == 0:
            logger.warning(f"No cell candidates for graph {graph_name} with conditions {name}.")
            continue

        candidate_nodes = pl.Series(cell_candidates.select("label")).to_list()
        selected_nodes = set(candidate_nodes)

        num_steps_to_search = 6
        # Find nodes that are first in certain period in the graph to be considered as t0 nodes
        G_ = copy.deepcopy(G)

        if prune_t0:
            for node in candidate_nodes:

                pred_nodes = graph_utils.get_all_predecessors(node, G_, max_level=num_steps_to_search)
                # If any predecessor node in the candidate nodes, then this node is not a t0 node
                if any(pred in candidate_nodes for pred in pred_nodes):
                    selected_nodes.remove(node)
                    continue

                if len(pred_nodes) == 0:
                    continue

                # Check that all predecessors are within radar distance limit
                pred_timestamps_ids = [(pd.Timestamp(p.split("_")[0]), int(p.split("_")[1])) for p in pred_nodes]
                pred_data = cell_data.filter(
                    pl.any_horizontal(
                        [
                            (pl.col("timestamp") == pl.lit(ts)) & (pl.col("identifier") == pl.lit(id_))
                            for ts, id_ in pred_timestamps_ids
                        ]
                    )
                )
                if pred_data.filter(pl.col("dist_from_radars.min") > pred_dist_limit).height > 0:
                    logger.warning(
                        f"Node {node} has predecessors that are too far from radars in graph {graph_name} with conditions {name}."
                    )
                    selected_nodes.remove(node)
                    continue

        # Get trajectories for the selected t0 nodes
        preds_succs = {}
        for n in selected_nodes:
            predecessors = graph_utils.get_predecessors(n, G, max_level=num_steps_to_hop)
            successors = graph_utils.get_successors(n, G, max_level=num_steps_to_hop)

            if "event" not in G.nodes[n]:
                G.nodes[n]["event"] = "t0_node."
            else:
                G.nodes[n]["event"] += "t0_node."

            preds_succs[n] = {
                "predecessors": predecessors,
                "successors": successors,
                "type": name,
            }
        if prune_duplicates:
            keep_midnodes = graph_utils.prune_overlapping_trajectories(
                preds_succs=preds_succs,
                graph=G,
            )
            keep_midnodes_all.extend(keep_midnodes)
        else:
            keep_midnodes_all.extend(list(preds_succs.keys()))
        tseries = graph_utils.get_timeseries(G, preds_succs, engine=engine, **config["subgraphs"])

        if tseries is None:
            logger.warning(f"No time series data for graph {graph_name} with conditions {name}.")
            continue

        tseries = tseries.with_columns(
            pl.col("timestamp").count().over("type", "t0_node", "level").alias("num_cells_at_level")
        )
        tseries = tseries.filter((pl.col("t0_node").is_in(keep_midnodes)))

        all_tseries.append(tseries)

    if len(all_tseries) == 0:
        logger.warning(f"No time series data for graph {graph_name} for cell conditions.")
        return

    # Concatenate all time series data
    tseries = pl.concat(all_tseries, how="diagonal_relaxed")
    if config["trajectory_selection"]["drop_database_values"]:
        drop_columns = [c for c in tseries.columns if "." in c]
        tseries = tseries.drop(drop_columns)

    logger.info(f"Processing graph {graph_name} with {tseries.height} time series entries from cell conditions.")

    # Write results to parquet files
    outpath = Path(config["output"]["storagepath"]) / config["output"]["intermediate_subfolder"]
    outpath.mkdir(parents=True, exist_ok=True)
    filename_all = config["output"]["all_trajectories_intermediate_filename"].format(
        graph_name=graph_name,
        type=typename,
    )
    tseries.write_parquet(outpath / filename_all)
    logger.info(f"Saved {tseries.height} rows to {outpath / filename_all}")

    return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config", type=str, help="Path to the config file")
    argparser.add_argument("startdate", type=str, help="Start date in YYYYMMDDHHMM format")
    argparser.add_argument("enddate", type=str, help="End date in YYYYMMDDHHMM format")
    argparser.add_argument(
        "--dbconf", type=str, default="config/database/database.yaml", help="Path to the database config file"
    )
    argparser.add_argument("-n", "--nworkers", type=int, default=1, help="Number of workers to use")
    argparser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    args = argparser.parse_args()

    # Parse start and end dates
    startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
    enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")

    # Read config file
    config = load_config(args.config)
    dbconf = load_config(args.dbconf)

    # Load cell conditions config
    cond_filepath = Path(config["trajectory_selection"]["cell_condition_filepath"])
    if cond_filepath.exists():
        with open(cond_filepath, "r") as f:
            config["trajectory_selection"]["select_cell_conditions"] = yaml.safe_load(f)
    else:
        logger.error(f"Cell condition config file {cond_filepath} does not exist.")
        sys.exit(1)

    min_duration = config["track_graphs"]["min_duration"]

    # Create database engine
    engine = get_engine(dbconf, pool_size=2 * args.nworkers, max_overflow=args.nworkers)

    # Get tracks from database
    with Session(engine) as session:
        graph_df = pl.read_database(
            query=GRAPH_SQL_QUERY.format(startdate=startdate, enddate=enddate, min_duration=min_duration),
            connection=session.bind,
        )

    # debug selection
    # python scripts/dataprocessing/select_trajectories.py config/plots/swiss-data/select_trajectories.yml 202107130235 202107131900 -n 1 -d

    if args.debug:
        graph_df = graph_df.filter(
            (pl.col("min_timestamp") == pd.Timestamp(startdate))
            & (pl.col("max_timestamp") == pd.Timestamp(enddate) - pd.Timedelta(minutes=5))
        )

    res = []
    # Get trajectory data for each graph
    for row in graph_df.iter_rows(named=True):

        if config["trajectory_selection"]["select_cell_conditions"]:
            res.append(dask.delayed(get_cell_condition_trajectory_data_for_graph)(row, config))

        if config["trajectory_selection"]["select_split_merges"]:
            # res.append(
            #     dask.delayed(get_split_merge_trajectory_data_for_graph)(
            #         row, config, prune_t0=True, prune_duplicates=True
            #     )
            # )
            # res.append(
            #     dask.delayed(get_split_merge_trajectory_data_for_graph)(
            #         row, config, prune_t0=True, prune_duplicates=False
            #     )
            # )
            # res.append(
            #     dask.delayed(get_split_merge_trajectory_data_for_graph)(
            #         row, config, prune_t0=False, prune_duplicates=True
            #     )
            # )
            res.append(
                dask.delayed(get_split_merge_trajectory_data_for_graph)(
                    row, config, prune_t0=False, prune_duplicates=False
                )
            )

    from dask.distributed import Client

    scheduler = "threads" if args.nworkers > 1 else "single-threaded"
    client = Client(n_workers=args.nworkers, threads_per_worker=1)

    if args.nworkers > 1:
        pbar = ProgressBar(dt=1)
        pbar.register()

    dataframes = dask.compute(res, scheduler=scheduler)

    # Concatenate dataframes
    # df = pl.concat([d[0] for d in dataframes[0] if d is not None], how="diagonal_relaxed")
    # df_filtered = pl.concat([d[1] for d in dataframes[0] if d is not None], how="diagonal_relaxed")

    # Read all parquet files from the intermediate subfolder
    inpath = Path(config["output"]["storagepath"]) / config["output"]["intermediate_subfolder"]
    all_trajectories_glob = config["output"]["all_trajectories_intermediate_filename"].format(graph_name="*", type="*")
    filtered_trajectories_glob = config["output"]["filtered_trajectories_intermediate_filename"].format(
        graph_name="*", type="*"
    )
    all_files = list(inpath.glob(all_trajectories_glob))
    filtered_files = list(inpath.glob(filtered_trajectories_glob))

    if len(all_files) == 0:
        logger.error(f"No trajectory files found in {inpath}.")
        sys.exit(1)

    if len(filtered_files) == 0:
        logger.error(f"No filtered trajectory files found in {inpath}.")
        sys.exit(1)

    df = pl.read_parquet(inpath / all_trajectories_glob)

    logging.info(f"Found {df.height} trajectories in the time range {startdate} - {enddate}.")
    logging.info("Saving trajectories to parquet files...")

    # Save to parquet files
    outpath = Path(config["output"]["storagepath"])
    outpath.mkdir(parents=True, exist_ok=True)

    save_interval = config["output"]["save_interval"]
    save_dateranges = pd.date_range(startdate, enddate, freq=save_interval)

    for start, end in zip(save_dateranges[:-1], save_dateranges[1:]):

        df_ = df.with_columns(t0_time=pl.col("t0_node").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).filter(
            (start <= pl.col("t0_time")) & (pl.col("t0_time") < end)
        )

        if df_.height == 0:
            logger.warning(f"No data for {start} - {end}.")
            continue

        filename = config["output"]["all_trajectories_filename"].format(
            startdate=start.strftime("%Y%m%d%H%M"), enddate=end.strftime("%Y%m%d%H%M")
        )
        df_.write_parquet(outpath / filename)
        logger.info(f"Saved {df_.height} rows to {outpath / filename}")

    # Copy config file to output directory
    shutil.copy(cond_filepath, outpath / "t0_filtering_conditions.yml")
    shutil.copy(args.config, outpath / Path(args.config).name)
