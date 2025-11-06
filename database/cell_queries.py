import geopandas as gpd
import pandas as pd
import numpy as np
import os
from sqlalchemy import select
from sqlalchemy.orm import Session
import networkx as nx
from networkx import DiGraph
from datetime import timedelta

from . import get_engine, Stormcells, NextCells


def load_stormcells_at_time(timestep, db_config):
    """Load storm cells at a specific timestep from the database.

    Parameters
    ----------
    timestep : datetime.datetime
        Timestep to load storm cells for.
    db_config : dict
        Database configuration.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing storm cells at the specified timestep.
    """
    engine = get_engine(db_config)
    with Session(engine) as session:
        query = select(Stormcells).filter(Stormcells.timestamp == timestep)
        cell_df = gpd.read_postgis(sql=query, con=session.bind, geom_col="geometry")
    return cell_df


def load_stormcells_between_times(starttime, endtime, db_config=None, cell_method=None):
    """Load storm cells between two timesteps from the database.

    Parameters
    ----------
    starttime : datetime.datetime
        Start timestep to load storm cells for.
    endtime : datetime.datetime
        End timestep to load storm cells for.
    db_config : dict
        Database configuration.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing storm cells between the specified timesteps.
    """
    engine = get_engine(db_config)
    with Session(engine) as session:
        query = select(Stormcells).filter(Stormcells.timestamp >= starttime, Stormcells.timestamp <= endtime)
        if cell_method:
            query = query.filter(Stormcells.method == cell_method)
        cell_df = gpd.read_postgis(sql=query, con=session.bind, geom_col="geometry")
    return cell_df


def load_storm_tracks(starttime, endtime, db_config, cell_method, timestep, only_single_tracks=True):
    """Query storm tracks in the given time range.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    db_conn : sqlalchemy.engine.base.Connection
        The database connection to use.
    cell_method : str
        The cell identification method.
    timestep : int
        Time step to use for the storm track (minutes).
    only_single_tracks : bool
        If True, only return tracks that have no splits or merges.

    Returns
    -------
    out : tuple
        Two-element tuple containing forward and backward storm tracks of type
        networkx.DiGraph.
    """
    forward_tracks = DiGraph()
    backward_tracks = DiGraph()

    # query cells
    cells = load_stormcells_between_times(
        starttime - timedelta(minutes=timestep),
        endtime + timedelta(minutes=timestep),
        db_config=db_config,
        cell_method=cell_method,
    )
    for _, cell in cells.iterrows():
        forward_tracks.add_node((cell.timestamp, cell.identifier))
        backward_tracks.add_node((cell.timestamp, cell.identifier))

    # query cell successors
    query_columns = [
        NextCells.timestamp,
        NextCells.identifier,
        NextCells.next_identifier,
    ]
    engine = get_engine(db_config)
    with Session(engine) as session:
        query = select(*query_columns).where(
            NextCells.timestamp >= (starttime - timedelta(minutes=timestep)),
            NextCells.timestamp <= endtime,
            NextCells.method == cell_method,
        )
        result = pd.read_sql(sql=query, con=session.bind)

    if len(result) == 0:
        raise ValueError(f"No storm tracks found in the given time range {starttime} - {endtime}.")

    for i, r in result.iterrows():

        next_time = r.timestamp + timedelta(minutes=timestep)
        node1 = (r.timestamp, r.identifier)
        node2 = (next_time, r.next_identifier)
        if not node1 in forward_tracks.nodes:
            raise KeyError(
                f"inconsistent forward connectivity information: source cell {node1[1]} at {node1[0]} does not exist"
            )
        if not node2 in forward_tracks.nodes:
            raise KeyError(
                f"inconsistent forward connectivity information: target cell {node2[1]} at {node2[0]} does not exist"
            )

        backward_tracks.add_edge(node2, node1)
        forward_tracks.add_edge(node1, node2)

    # Build tracks as separate pandas dataframes
    timestamps = sorted(cells.timestamp.unique().to_pydatetime().tolist())

    cells["parent_track_ids"] = pd.Series(dtype="object")
    cells["child_track_ids"] = pd.Series(dtype="object")
    track_ids = {}
    id_counter = 0
    # Initialize tracks at first timestamp
    for i, cell in cells[cells.timestamp == timestamps[0]].iterrows():

        track_ids[(cell.timestamp, cell.identifier)] = [
            id_counter,
        ]
        cells.at[i, "parent_track_ids"] = [
            id_counter,
        ]
        id_counter += 1

    # Get the track for each cell at first timestamp
    for tt in timestamps[1:]:
        cells_at_time = cells[cells.timestamp == tt]

        for i, cell in cells_at_time.iterrows():
            node = (cell.timestamp, cell.identifier)

            parents = list(forward_tracks.predecessors(node))
            if len(parents) == 0:
                # Start a new track
                cells.at[i, "parent_track_ids"] = [
                    id_counter,
                ]
                track_ids[node] = [
                    id_counter,
                ]
                id_counter += 1
                continue

            # If multiple parents -> cell is merger
            # if len(parents) > 1:
            ids = []
            for parent in parents:
                ids.extend(track_ids[parent])
            track_ids[node] = ids
            cells.at[i, "parent_track_ids"] = np.unique(ids).tolist()

    # Find child track ids
    for tt in timestamps:
        cells_at_time = cells[cells.timestamp == tt]

        for i, cell in cells_at_time.iterrows():
            node = (cell.timestamp, cell.identifier)

            children = list(backward_tracks.predecessors(node))

            if len(children) == 0:
                cells.at[i, "child_track_ids"] = [
                    np.nan,
                ]
                continue

            ids = []
            # Multiple children -> cell splits
            for child in children:
                ids.extend(track_ids[child])
            cells.at[i, "child_track_ids"] = np.unique(ids).tolist()

    # Create dataframes for each track
    tracks = {}
    single_tracks = []
    for track_id in range(id_counter):
        track_parents = cells[cells.parent_track_ids.apply(lambda x: track_id in x)]
        track_children = cells[cells.child_track_ids.apply(lambda x: track_id in x)]
        track = pd.concat([track_parents, track_children]).drop_duplicates(subset=["timestamp", "identifier"])
        track = track.sort_values("timestamp")
        # track = track[track.child_track_ids.apply(lambda x: len(x) == 0)]
        tracks[track_id] = track

        if (
            track.parent_track_ids.apply(lambda x: len(x)).max() == 1
            and track.child_track_ids.apply(lambda x: len(x)).max() == 1
        ):
            single_tracks.append(track)

    if only_single_tracks:
        # Remove tracks that have multiple parent_track_ids or child_track_ids
        return single_tracks

    return tracks, forward_tracks, backward_tracks
