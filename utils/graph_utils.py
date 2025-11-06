"""Utils for graph operations."""

from ast import literal_eval
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path
import polars as pl
from sqlalchemy import create_engine, select, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.schema import MetaData
import seaborn as sns
from collections import defaultdict, Counter
from itertools import chain
import geopandas as gpd

from . import data_utils, plot_utils


def string_decoder(rep):
    """Decode strings in the GML definitions."""
    # Datetime instances
    try:
        return datetime.strptime(rep, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        if isinstance(rep, str):
            orig_rep = rep
            try:
                return literal_eval(rep)
            except SyntaxError as err:
                raise ValueError(f"{orig_rep!r} is not a valid Python literal") from err
        else:
            raise ValueError(f"{rep!r} is not a string")


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


def plot_graph(
    G,
    graph_name,
    pos=None,
    time_resolution="30min",
    outpath=Path("."),
    ext="pdf",
    color_var=None,
    color_cmap=None,
    color_norm=None,
    edge_labels=None,
    highlight_nodes=None,
    highlight_color="red",
    highlight_node_edgewidth=1.5,
    fill_color="#303193",
    edgecolor="#303193",
    fig=None,
    axs=None,
    savefig_kwargs=dict(dpi=600, bbox_inches="tight"),
    savefig=True,
    write_title=True,
    label_key="identifier",
    xaxis_pad=0.3,
):
    """Plot the graph with timestamps on the x-axis."""
    timestamps = [n["timestamp"] for _, n in G.nodes(data=True)]
    timestamps = pd.to_datetime(timestamps)
    timestamps = pd.Series(timestamps).sort_values()

    xmin = timestamps.min()
    xmax = timestamps.max()

    num_timesteps = len(timestamps)
    num_time_resolutions = int((xmax - xmin) / pd.Timedelta(time_resolution))

    if fig is None:
        fig = plt.figure(figsize=(1 * num_time_resolutions, 5))
    if axs is None:
        axs = fig.add_subplot(111)

    labels = {n: G.nodes[n][label_key] for n in G}

    if pos is None:
        pos = nx.multipartite_layout(G, subset_key="timestamp")

    min_x_pos = min([pos[i][0] for i in pos])
    max_x_pos = max([pos[i][0] for i in pos])

    xtick_labels = [
        xmin,
        *pd.date_range(xmin.ceil(time_resolution), xmax.floor(time_resolution), freq=time_resolution).to_list(),
        xmax,
    ]
    xtick_labels_str = [s.strftime("%H:%M") for s in xtick_labels]

    xtick_locs = [min_x_pos + ((i - xmin) / (xmax - xmin)) * (max_x_pos - min_x_pos) for i in xtick_labels]

    if color_var is not None:
        # Get color values from the graph nodes
        color_values = [G.nodes[n][color_var] for n in G.nodes]

        colors = [color_cmap(color_norm(v)) if np.isfinite(v) else "k" for v in color_values]
    else:
        colors = [fill_color] * G.number_of_nodes()  # Default color for nodes

    if highlight_nodes is not None:
        # Highlight specific nodes
        highlight_nodes = set(highlight_nodes)
        edgecolors = [colors[i] if n not in highlight_nodes else highlight_color for i, n in enumerate(G.nodes)]
        linewidths = [highlight_node_edgewidth if n in highlight_nodes else 0.5 for n in G.nodes]
    else:
        edgecolors = edgecolor
        linewidths = None

    nx.draw(
        G,
        pos,
        node_size=250,
        labels=labels,
        with_labels=True,
        node_color=colors,
        edge_color="k",
        alpha=1.0,
        ax=axs,
        hide_ticks=False,
        horizontalalignment="center",
        verticalalignment="center",
        font_size="small",
        font_color="white",
        clip_on=False,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_color="black",
            font_size="small",
            ax=axs,
            rotate=False,
            label_pos=0.4,
            clip_on=False,
        )
    axs.axis("on")  # turns on axis
    axs.xaxis.set_ticks(xtick_locs, xtick_labels_str, rotation=25, ha="right")
    axs.grid(axis="x", color="gray", linestyle="--", linewidth=0.5)
    axs.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    # remove spines on left and top
    axs.spines["left"].set_visible(False)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.set_xlim(min_x_pos - xaxis_pad, max_x_pos + xaxis_pad)
    if write_title:
        axs.set_title(
            f"Cell track with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n"
            f"from {xmin.strftime('%Y-%m-%d %H:%M')} to {xmax.strftime('%Y-%m-%d %H:%M')}"
        )

    if savefig:
        fig.tight_layout()
        if isinstance(ext, str):
            ext = [ext]

        for e in ext:
            fig.savefig(
                outpath / f"track_graph_{graph_name}.{e}",
                **savefig_kwargs,
            )
        plt.close(fig)
        del fig
    else:
        return fig, axs


def get_cells_in_graph(graph, engine, dbconf):
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="raincells", views=False)
    table = metadata.tables["raincells.stormcells"]

    keys = [(graph.nodes[n]["timestamp"], graph.nodes[n]["identifier"], graph.nodes[n]["method"]) for n in graph.nodes]

    query = select(
        table.c.timestamp,
        table.c.identifier,
        table.c.method,
        table.c.geometry,
    ).where(
        or_(and_(table.c.timestamp == t, table.c.identifier == i, table.c.method == m) for t, i, m in keys),
    )

    with Session(engine) as session:
        df = gpd.read_postgis(query, session.bind, geom_col="geometry", index_col=None, crs=dbconf["projection"])

    return df


def get_cells(starttime, endtime, method, engine, dbconf):
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="raincells", views=False)
    table = metadata.tables["raincells.stormcells"]

    query = select(
        table.c.timestamp,
        table.c.identifier,
        table.c.method,
        table.c.geometry,
    ).where(
        and_(
            table.c.timestamp >= starttime,
            table.c.timestamp <= endtime,
            table.c.method == method,
        )
    )

    with Session(engine) as session:
        df = gpd.read_postgis(query, session.bind, geom_col="geometry", index_col=None, crs=dbconf["projection"])

    return df


def plot_graph_snapshots(
    graph,
    t0_time,
    fig,
    axs,
    cell_data,
    plot_conf,
    engine,
    dbconf,
    n_images=6,
    plot_times=None,
    bumber_km=20,
    plot_var="RATE",
    cell_plot_kwargs=None,
    track_color="C0",
    plot_cbar=True,
    annotate=True,
    time_title=True,
    annotate_key="identifier",
    annotate_cells="all",  # or "graph"
    plot_motion_from_rate=False,
    sparse_factor=6,
):
    """Plot snapshots of the graph at different time steps."""

    # Load data
    if plot_motion_from_rate:
        if plot_var != "RATE":
            plot_vars = [plot_var, "RATE"]
        else:
            plot_vars = [plot_var]

        add_to_timesteps = 1

    else:
        plot_vars = [plot_var]
        add_to_timesteps = 0

    if plot_times is None:
        get_times = (
            pd.date_range(
                start=t0_time - timedelta(minutes=n_images * 5),
                periods=(2 * n_images + 1 + add_to_timesteps),
                freq=plot_conf.freq,
            )
            .to_pydatetime()
            .tolist()
        )
    else:
        get_times = plot_times

        if plot_motion_from_rate:
            get_times = get_times + [get_times[-1] + pd.Timedelta(plot_conf.freq)]

    input_data_conf = {k: v for k, v in plot_conf.input_data.items() if k in plot_vars}
    dataset = data_utils.load_data(
        input_data_conf,
        get_times[0],
        get_times,
        len(get_times),
        None,
    )

    if plot_motion_from_rate:
        get_times = get_times[:-1]  # Remove last time, as it is only used for motion calculation

    # Get cells in the graph
    graph_cells = get_cells_in_graph(graph, engine, dbconf)

    # Calculate zoom bbox
    if isinstance(bumber_km, (int, float)):
        bumber_km = (bumber_km, bumber_km, bumber_km, bumber_km)
    minx, miny, maxx, maxy = graph_cells.total_bounds
    min_col = np.round(((minx - dataset.x.values.min()) / 1000 - bumber_km[0]) / dataset.x.values.size, 3).item()
    max_col = np.round(((maxx - dataset.x.values.min()) / 1000 + bumber_km[1]) / dataset.x.values.size, 3).item()
    min_row = np.round(((miny - dataset.y.values.min()) / 1000 - bumber_km[2]) / dataset.y.values.size, 3).item()
    max_row = np.round(((maxy - dataset.y.values.min()) / 1000 + bumber_km[3]) / dataset.y.values.size, 3).item()

    bbox = [min_col, max_col, min_row, max_row]
    len_x = dataset.x.values.size
    len_y = dataset.y.values.size
    print(int(min_col * len_x), int(max_col * len_x), int(min_row * len_y), int(max_row * len_y))

    track_plot_kwargs = cell_plot_kwargs.copy()
    track_plot_kwargs["edgecolor"] = track_color

    var_name = input_data_conf[plot_var].variable

    for i, t in enumerate(get_times):
        print(f"Plotting time {t} ({i+1}/{len(get_times)})")
        ax = axs.flatten()[i]
        if time_title:
            ax.set_title(t.strftime("%Y-%m-%d %H:%M:%S"), fontsize="x-small")

        im = dataset.sel(time=t)[var_name].to_numpy().squeeze()

        if plot_motion_from_rate:
            cur_rate_im = dataset.sel(time=t)["RATE"].to_numpy().squeeze()
            next_rate_im = dataset.sel(time=t + pd.Timedelta(plot_conf.freq))["RATE"].to_numpy().squeeze()

            # Calculate motion vectors from rate images
            motion = data_utils.compute_advection_field(
                cur_rate_im,
                next_rate_im,
                pyr_scale=0.5,
                levels=6,
                winsize=200,
                iterations=10,
                poly_n=7,
                poly_sigma=1.5,
                filter_stddev=0,
                minval=0,
            )
            # sparse_factor = 6
            xx = dataset.x.values[int(min_col * len_x) : int(max_col * len_x) + 1 : sparse_factor]
            yy = dataset.y.values[int(min_row * len_y) : int(max_row * len_y) + 1 : sparse_factor]
            mu = motion[
                int(min_row * len_y) : int(max_row * len_y) + 1 : sparse_factor,
                int(min_col * len_x) : int(max_col * len_x) + 1 : sparse_factor,
                0,
            ]
            mv = motion[
                int(min_row * len_y) : int(max_row * len_y) + 1 : sparse_factor,
                int(min_col * len_x) : int(max_col * len_x) + 1 : sparse_factor,
                1,
            ]
            print(xx.shape, yy.shape, mu.shape, mv.shape)
            ax.quiver(
                xx,
                yy,
                mu,
                mv,
                # scale=0.0005,
                scale=0.001 / (sparse_factor / 3),
                angles="xy",
                scale_units="xy",
                zorder=0,
                color="k",
                linewidth=0.5,
                alpha=0.7,
                # rasterized=True,
            )
            # ax.streamplot(
            #     dataset.x.values,
            #     dataset.y.values,
            #     motion[:, :, 0],
            #     motion[:, :, 1],
            #     density=5,
            #     zorder=0,
            #     color="k",
            #     linewidth=0.5,
            #     broken_streamlines=False,
            # )

        nan_mask = dataset.sel(time=t)[f"{var_name}_nan_mask"].values.squeeze()
        zero_mask = np.isclose(im, 0)

        im[zero_mask] = np.nan
        im[nan_mask] = np.nan

        cbar = plot_utils.plot_array(
            ax,
            im.copy(),
            x=dataset.x.values,
            y=dataset.y.values,
            qty=input_data_conf[plot_var].cmap_qty,
            colorbar=False,
            zorder=1,
        )

        # Get all cells at this time
        cells_at_this_time = get_cells(t, t, cell_data.select("method").unique().item(), engine, dbconf)
        cells_in_graph = pd.merge(
            cells_at_this_time, cell_data.to_pandas(), on=["timestamp", "identifier", "method"], how="inner"
        )

        # First plot all cells
        cells_at_this_time.plot(
            ax=ax,
            zorder=20,
            autolim=False,
            **cell_plot_kwargs,
        )
        # highlight cells that are in the graph
        cells_in_graph.plot(
            ax=ax,
            zorder=20,
            autolim=False,
            **track_plot_kwargs,
        )
        if annotate:
            if annotate_cells == "all":
                cells_to_annotate = cells_at_this_time
            elif annotate_cells == "graph":
                cells_to_annotate = cells_in_graph
            else:
                raise ValueError(f"Invalid value for annotate_cells: {annotate_cells}, should be 'all' or 'graph'")
            # annotate the cells with identifier
            for i, row in cells_to_annotate.iterrows():
                # print(row)
                cell_bbox = row.geometry.bounds
                label = row[annotate_key]
                ax.annotate(
                    text=f"{label}",
                    xy=(cell_bbox[2], cell_bbox[3]),
                    # (row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize="x-small",
                    color="k",
                    ha="center",
                    va="center",
                    zorder=25,
                    # xytext=(cell_bbox[2], cell_bbox[3]),
                    # textcoords="offset points",
                    bbox=dict(facecolor="white", edgecolor="white", pad=0.1, alpha=0.8, boxstyle="round"),
                )

    for ax in axs.flatten()[: len(get_times)]:
        ax.set_xticks(
            np.arange(
                dataset.x.values.min(),
                dataset.x.values.max(),
                plot_conf.tick_spacing * 1e3,
            )
        )
        ax.set_yticks(
            np.arange(
                dataset.y.values.min(),
                dataset.y.values.max(),
                plot_conf.tick_spacing * 1e3,
            )
        )
        ax.set_aspect(1)

        im_width = dataset.x.values.max() - dataset.x.values.min()
        im_height = dataset.y.values.max() - dataset.y.values.min()

        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())

        ax.grid(lw=0.8, color="tab:gray", ls=":", zorder=11)

        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        ax.set_xlim(
            (
                dataset.x.values.min() + im_width * bbox[0],
                dataset.x.values.min() + im_width * bbox[1],
            )
        )
        ax.set_ylim(
            (
                dataset.y.values.min() + im_height * bbox[2],
                dataset.y.values.min() + im_height * bbox[3],
            )
        )

        # ax.axis("off")

    # Plot colorbar in place of last axis
    if plot_cbar:
        for ax in axs.flatten()[len(get_times) :]:
            ax.axis("off")
        cbar = plot_utils.plot_colorbar(ax, input_data_conf[plot_var].cmap_qty)

    return fig, axs, cbar


def plot_graph_timeseries(
    graph,
    cell_data,
    variable,
    ax,
    linecolor="C0",
    marker="o",
    markersize=None,
    annotate=True,
    annotate_key="identifier",
    majority_variable="area",
    plot_majority_path=True,
    linewidth=1,
    majority_linewidth=2,
):
    linestyle = "--" if plot_majority_path else "-"
    for node in graph.nodes:
        timestamp = graph.nodes[node]["timestamp"]
        identifier = graph.nodes[node]["identifier"]
        method = graph.nodes[node]["method"]

        node_data = (
            cell_data.filter(
                (pl.col("timestamp") == timestamp) & (pl.col("identifier") == identifier) & (pl.col("method") == method)
            )
            .to_pandas()
            .iloc[0]
        )
        ax.scatter(
            timestamp, node_data[variable], marker=marker, color=linecolor, s=markersize, label=f"{node}", zorder=10
        )
        if annotate and node_data[variable] is not None and graph.nodes[node][annotate_key] is not None:
            ax.annotate(
                f"{graph.nodes[node][annotate_key]}",
                (timestamp, node_data[variable]),
                fontsize="large",
                color="k",
                ha="left",
                va="top",
                zorder=15,
            )

        successors_majority_values = []
        successors_majority_nodes = []
        successors_datavalues = []
        for successor in graph.successors(node):
            successor_data = (
                cell_data.filter(
                    (pl.col("timestamp") == graph.nodes[successor]["timestamp"])
                    & (pl.col("identifier") == graph.nodes[successor]["identifier"])
                    & (pl.col("method") == graph.nodes[successor]["method"])
                )
                .to_pandas()
                .iloc[0]
            )
            successors_majority_values.append(successor_data[majority_variable])
            successors_majority_nodes.append(successor)
            successors_datavalues.append(successor_data[variable])

            ax.plot(
                [timestamp, graph.nodes[successor]["timestamp"]],
                [node_data[variable], successor_data[variable]],
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=5,
            )

    if plot_majority_path:
        # At each timestep, if there is a split or merge in the graph,
        # plot the edge between the largest cell at that timestep and largest cell at the next timestep
        timestamps = sorted(set(graph.nodes[n]["timestamp"] for n in graph.nodes))
        prev_node = None
        for cur_step in timestamps:

            if prev_node is not None:
                cur_nodes = list(graph.successors(prev_node))
                # print(f"Successors of {prev_node} at {cur_step}: {cur_nodes}")
            else:
                cur_nodes = [n for n in graph.nodes if graph.nodes[n]["timestamp"] == cur_step]

            if len(cur_nodes) == 0:
                # print(f"No possible nodes found at {cur_step}, stopping.")
                break
            else:
                # Pick cell with the largest area at this timestep
                cur_ids = [graph.nodes[n]["identifier"] for n in cur_nodes]
                cur_data = (
                    cell_data.filter((pl.col("timestamp") == cur_step) & (pl.col("identifier").is_in(cur_ids)))
                    .sort(by=majority_variable, descending=True)
                    .head(1)
                    .to_pandas()
                    .iloc[0]
                )
                cur_node = f"{cur_data['timestamp']:%Y-%m-%dT%H:%M:%S}_{cur_data['identifier']}"

            # print(f"Current node: {cur_node} at {cur_step}")

            if prev_node is not None:
                prev_step = timestamps[timestamps.index(cur_step) - 1]
                prev_data = (
                    cell_data.filter(
                        (pl.col("timestamp") == prev_step) & (pl.col("identifier") == int(prev_node.split("_")[1]))
                    )
                    .unique()
                    .head(1)
                    .to_pandas()
                    .iloc[0]
                )
                # print(cur_node, cur_data, prev_node, prev_data)
                if cur_data[variable] and prev_data[variable]:
                    # Plot the edge between the previous node and current node
                    ax.plot(
                        [prev_step, cur_step],
                        [prev_data[variable], cur_data[variable]],
                        color=linecolor,
                        linewidth=majority_linewidth,
                        linestyle="-",
                        zorder=6,
                    )

            prev_node = cur_node


def build_filter_from_conditions(conditions: dict):
    """Build a polars filter expression from conditions."""

    min_conditions = [(pl.col(k) >= v["min"]) for k, v in conditions.items() if "min" in v.keys()]
    max_conditions = [(pl.col(k) < v["max"]) for k, v in conditions.items() if "max" in v.keys()]
    eq_conditions = [(pl.col(k) == v["eq"]) for k, v in conditions.items() if "eq" in v.keys()]
    null_conditions = [(pl.col(k).is_null()) for k, v in conditions.items() if "isnull" in v.keys() and v["isnull"]]
    not_null_conditions = [
        (pl.col(k).is_not_null()) for k, v in conditions.items() if "isnull" in v.keys() and not v["isnull"]
    ]
    # Combine all conditions
    conditions = min_conditions + max_conditions + eq_conditions + null_conditions + not_null_conditions
    return conditions


def calculate_diff_variables(trajectories, data, sum_vars=[], max_vars=[]):
    """
    For each trajectory, calculate the difference of specified variables,
    summed over the cells in trajectory at some time, to the t0 node values.
    """
    diff_vars = list(set(sum_vars + max_vars))

    trajectories = trajectories.join(data, on=["timestamp", "identifier", "method"], how="left").unique()

    # Get t0 node values
    t0_nodes = (
        trajectories.filter(pl.col("level") == 0)
        .select("type", "t0_node", "level", "num_cells_at_level", *diff_vars)
        .with_columns((pl.col(v).alias(f"t0_{v}")) for v in diff_vars)
        .select("type", "t0_node", "level", "num_cells_at_level", *[f"t0_{v}" for v in diff_vars])
        .unique()
    )

    trajectory_sums = (
        trajectories.group_by(["type", "t0_node", "level"])
        # .sum()
        .agg(
            pl.col("num_cells_at_level").min().alias("num_cells_at_level"),
            pl.col("timestamp").min().alias("timestamp"),
            *[pl.col(v).sum().alias(v) for v in sum_vars],
            *[pl.col(v).max().alias(v) for v in max_vars],
        )
        .join(
            t0_nodes,
            on=["type", "t0_node"],
            how="inner",
        )
        .with_columns([(pl.col(v) / pl.col(f"t0_{v}") - 1).alias(f"t0_reldiff:{v}") for v in diff_vars])
        .with_columns([(pl.col(v) - pl.col(f"t0_{v}")).alias(f"t0_absdiff:{v}") for v in diff_vars])
        .select(
            "type",
            "timestamp",
            "t0_node",
            "level",
            "num_cells_at_level",
            *diff_vars,
            *[f"t0_reldiff:{v}" for v in diff_vars],
            *[f"t0_absdiff:{v}" for v in diff_vars],
            *[pl.col(f"t0_{v}") for v in diff_vars],
        )
        .with_columns([(pl.col(f"t0_reldiff:{v}") * 100).alias(f"t0_reldiff:{v}:pct") for v in diff_vars])
    )
    return trajectory_sums


def get_all_predecessors(node, G, max_level=10):
    """Get all predecessors of the node in the graph up to max_level_steps."""
    predecessors = set()
    current_level_nodes = {node}
    for level in range(max_level):
        next_level_nodes = set()
        for n in current_level_nodes:
            preds = set(G.predecessors(n))
            next_level_nodes.update(preds)
            predecessors.update(preds)
        if not next_level_nodes:
            break
        current_level_nodes = next_level_nodes
    return sorted(predecessors, reverse=True)


def get_all_successors(node, G, max_level=10):
    """Get all successors of the node in the graph up to max_level_steps."""
    successors = set()
    current_level_nodes = {node}
    for level in range(max_level):
        next_level_nodes = set()
        for n in current_level_nodes:
            succs = set(G.successors(n))
            next_level_nodes.update(succs)
            successors.update(succs)
        if not next_level_nodes:
            break
        current_level_nodes = next_level_nodes
    return sorted(successors)


def get_predecessors(n, G, level=0, max_level=10):
    """Get predecessors of a node in the graph."""
    # plot_graph(G, "test_graph", time_resolution="5min", outpath=Path("."), ext=[ "png"])
    selected_predecessors_by_level = defaultdict(set)
    candidate_predecessors_by_level = defaultdict(set)
    candidate_predecessors_by_level[1] = set(G.predecessors(n))

    for level in range(1, max_level + 1):
        stop_after_this_level = False
        stop_at_this_level = False
        for p in candidate_predecessors_by_level[level]:
            event = ""
            n_in_edges = G.in_degree(p)
            n_out_edges = G.out_degree(p)

            included = True
            if n_out_edges > 1:
                # This node is a split, we want to include it in the graph only if
                # all of its successors are included
                splitting_nodes = list(G.successors(p))
                num_nodes = len(splitting_nodes)
                if not all([n in selected_predecessors_by_level[level - 1] for n in splitting_nodes]):
                    # If not all successors are included, we stop at this level
                    # and do not include this node
                    stop_at_this_level = True
                    included = False
                else:
                    event += f"splits_into_{num_nodes}."
            if n_in_edges == 0:
                # This node was born, so we don't want to go further back in the graph
                # in any branch
                stop_after_this_level = True
                included = True
                event += f"born."

            if n_in_edges > 1:
                event += f"merged_from_{n_in_edges}."

            if included:
                selected_predecessors_by_level[level].add(p)
                candidate_predecessors_by_level[level + 1].update(G.predecessors(p))

            if not event:
                event = "simple."
            G.nodes[p]["event"] = event

        if stop_at_this_level:
            # We dont want to include anything at this level, so we stop here
            # and remove all predecessors at this level
            selected_predecessors_by_level[level] = set()
            break
        if stop_after_this_level:
            break

    # If no predecessors were found, we return an empty list
    if len(selected_predecessors_by_level) == 0:
        return []

    # Get all predecessors from all levels and sort
    predecessors = sorted(list(set.union(*[v for k, v in selected_predecessors_by_level.items()])), reverse=True)
    return predecessors


def get_successors(n, G, level=0, max_level=10):
    """Get predecessors of a node in the graph."""
    # plot_graph(G, "test_graph", time_resolution="5min", outpath=Path("."), ext=[ "png"])
    selected_successors_by_level = defaultdict(set)

    candidate_successors_by_level = defaultdict(set)
    candidate_successors_by_level[1] = set(G.successors(n))

    for level in range(1, max_level + 1):
        stop_at_this_level = False
        for p in candidate_successors_by_level[level]:
            event = ""
            n_in_edges = G.in_degree(p)
            n_out_edges = G.out_degree(p)

            included = True
            if n_in_edges > 1:
                # This node is a merge, so want to include it in the graph only if
                # all of its predecessors are included
                merging_nodes = list(G.predecessors(p))
                if not all([n in selected_successors_by_level[level - 1] for n in merging_nodes]):
                    stop_at_this_level = True
                    included = False
                else:
                    event += f"merged_from_{len(merging_nodes)}."

            if n_out_edges == 0:
                event += f"died."

            if n_out_edges > 1:
                event += f"splits_into_{n_out_edges}."

            if included:
                selected_successors_by_level[level].add(p)
                candidate_successors_by_level[level + 1].update(G.successors(p))
            if not event:
                event = "simple"
            G.nodes[p]["event"] = event

        if stop_at_this_level:
            # We dont want to include anything at this level, so we stop here
            # and remove all successors at this level
            selected_successors_by_level[level] = set()
            break

    # If no successors were found, we return an empty list
    if len(selected_successors_by_level) == 0:
        return []
    # Get all successors from all levels and sort
    successors = sorted(list(set.union(*[v for _, v in selected_successors_by_level.items()])))
    return successors


def get_predecessors_recursive(n, G, level=0, max_level=3):
    if level >= max_level:
        return []
    predecessors = list(G.predecessors(n))
    if not predecessors:
        return []
    return predecessors + [p for pred in predecessors for p in get_predecessors(pred, G, level + 1, max_level)]


def get_successors_recursive(n, G, level=0, max_level=3):
    if level >= max_level:
        return []
    successors = list(G.successors(n))
    if not successors:
        return []
    # import ipdb

    # ipdb.set_trace()
    # Check if there are any merges in the successors
    return successors + [s for succ in successors for s in get_successors(succ, G, level + 1, max_level)]


def prune_overlapping_trajectories(preds_succs, graph, size_type="num_nodes_total"):
    """Prune overlapping trajectories in the graph.

    Parameters
    ----------
    preds_succs : dict
        Dictionary with predecessors and successors for each node.
    graph : nx.Graph
        The graph to prune.
    max_overlap : int, optional
        Maximum number of overlapping nodes allowed, by default 3.

    Returns
    -------
    list
        Midnodes to keep in the graph.
    """
    counts_preds = {k: Counter(v["predecessors"]) for k, v in preds_succs.items()}
    counts_succs = {k: Counter(v["successors"]) for k, v in preds_succs.items()}

    all_succs = sorted(set(chain.from_iterable([v["successors"] for v in preds_succs.values()])))
    all_preds = sorted(set(chain.from_iterable([v["predecessors"] for v in preds_succs.values()])))

    # Calculate durations for each midnode
    # and store them in the preds_succs dictionary
    for midnode, data in preds_succs.items():
        trajectory_min = min([pd.Timestamp(s.split("_")[0]) for s in data["predecessors"] + [midnode]])
        trajectory_max = max([pd.Timestamp(s.split("_")[0]) for s in data["successors"] + [midnode]])
        preds_succs[midnode]["duration"] = trajectory_max - trajectory_min

    prune_midnodes_succs = set()
    overlapping_midnodes = {}
    reason_for_pruning = defaultdict(list)
    # Iterate through each possible successor node and find in how many branches it appears
    for succ in all_succs:
        midnodes = set([k for k, v in counts_succs.items() if v[succ] > 0])
        midnodes_list = sorted(midnodes)

        if len(midnodes_list) > 1:
            # If the successor appears in more than one branch, we need to prune the branches
            longest_branch = _select_longest_branch(preds_succs, midnodes_list, graph, mode="successors")

            # Mark all other midnodes containing this node for pruning
            prune_midnodes_succs.update(midnodes - set([longest_branch]))
            overlapping_midnodes[succ] = midnodes
            for midnode in midnodes - set([longest_branch]):
                reason_for_pruning[midnode].append(succ)

    keep_midnodes_succs = set(preds_succs.keys()) - prune_midnodes_succs

    prune_midnodes_preds = set()
    # Iterate through each possible predecessor node and find in how many branches it appears
    for pred in all_preds:
        midnodes = set([k for k, v in counts_preds.items() if v[pred] > 0])
        midnodes_list = sorted(midnodes)

        if len(midnodes) > 1:
            longest_branch = _select_longest_branch(preds_succs, midnodes_list, graph, mode="predecessors")

            # Mark all other midnodes containing this node for pruning
            prune_midnodes_preds.update(midnodes - set([longest_branch]))
            overlapping_midnodes[pred] = midnodes
            for midnode in midnodes - set([longest_branch]):
                reason_for_pruning[midnode].append(pred)

    keep_midnodes_preds = set(preds_succs.keys()) - prune_midnodes_preds
    # Now we have two sets of midnodes to keep, one for successors and one for predecessors
    keep_midnodes = keep_midnodes_succs.intersection(keep_midnodes_preds)
    pruned_midnodes = set(preds_succs.keys()) - keep_midnodes

    # if size_type == "duration" and "2021-07-13T17:40:00_2" in pruned_midnodes:
    #     import ipdb

    #     ipdb.set_trace()

    # if "2021-07-13T17:40:00_2" in pruned_midnodes:
    #     import ipdb

    #     ipdb.set_trace()

    return keep_midnodes


def _select_longest_branch(preds_succs, midnodes_list, graph, mode="successors"):
    """Select the longest branch from the list of midnodes.

    Selection criteria:
    1. Branch with the longest duration.
    2. If multiple branches have the same duration, select the one with the most nodes.
    3. If multiple branches have the same number of nodes, select the one with the most successors/predecessors (depending on mode).
    4. If still tied, select randomly.

    """
    # Debug variable
    branch_sizes_num_nodes = [len(preds_succs[x]["successors"] + preds_succs[x]["predecessors"]) for x in midnodes_list]
    branch_durations = [preds_succs[x]["duration"] for x in midnodes_list]
    if Counter(branch_durations)[max(branch_durations)] > 1:
        # If there are multiple branches with the same duration
        # we need to find the longest branch by number of nodes
        longest_branches = [
            midnodes_list[i] for i, _ in enumerate(midnodes_list) if branch_durations[i] == max(branch_durations)
        ]

        longest_branches_sizes = [
            len(preds_succs[x]["successors"] + preds_succs[x]["predecessors"]) for x in longest_branches
        ]
        if Counter(longest_branches_sizes)[max(longest_branches_sizes)] > 1:
            # If multiple branches have the same number of nodes,
            # pick the one that has the most nodes in successors/predecessors (depending on mode)
            len_mode = [len(preds_succs[x][mode]) for x in longest_branches]
            if Counter(len_mode)[max(len_mode)] > 1:
                # If multiple branches have the same number of successors,
                # finally pick the longest branch randomly
                longest_branch = np.random.choice(
                    [longest_branches[i] for i, s in enumerate(longest_branches) if len_mode[i]]
                )
                graph.nodes[longest_branch]["event"] += "longest_branch_randomly_chosen."
            else:
                longest_branch = max(longest_branches, key=lambda x: len(preds_succs[x][mode]))
                graph.nodes[longest_branch]["event"] += f"longest_branch_chosen_by_{mode}_size."
        else:
            # Multiple branches have the same duration, but only one has the most nodes
            longest_branch = longest_branches[longest_branches_sizes.index(max(longest_branches_sizes))]
            graph.nodes[longest_branch]["event"] += "longest_branch_chosen_by_num_nodes."
    else:
        # If there is only one branch with the longest duration, we can return it directly
        longest_branch = max(midnodes_list, key=lambda x: preds_succs[x]["duration"])
    graph.nodes[longest_branch]["event"] += "longest_branch_chosen_by_duration."
    return longest_branch


def get_cell_data(graph, quantities, statistics, engine):
    """Get rasterstats data for the cells in the graph."""
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="raincells", views=True)
    stat_view = metadata.tables["raincells.raincells_with_statistics"]

    keys = []
    for node in graph.nodes:
        data = graph.nodes[node]
        method = data.get("method", "unknown")
        timestamp = data.get("timestamp", "unknown")
        identifier = data.get("identifier", "unknown")
        keys.append((timestamp, identifier, method))

    # Build query
    query = select(
        stat_view.c.timestamp,
        stat_view.c.identifier,
        stat_view.c.method,
        stat_view.c.quantity,
        stat_view.c.statistic,
        stat_view.c.value,
        stat_view.c.area,
    ).where(
        and_(
            stat_view.c.quantity.in_(quantities),
            stat_view.c.statistic.in_(statistics),
            or_(
                and_(stat_view.c.timestamp == t, stat_view.c.identifier == i, stat_view.c.method == m)
                for t, i, m in keys
            ),
        )
    )

    with Session(engine) as session:
        # sql queryn tulostus arvojen kanssa
        # print(str(query.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
        df_data = pl.read_database(query=query, connection=session.bind)

    df = df_data.with_columns(pl.format("{}.{}", "quantity", "statistic").alias("on")).pivot(
        on="on",
        # index=["type", "t0_node", "timestamp", "level", "identifier", "method", "area", "event"],
        index=set(df_data.columns) - set(["quantity", "statistic", "value"]),
        values=["value"],
    )

    # if some quantity.statistic is not present in the data, we need to fill it with NaN
    all_quantities = set(quantities)
    all_statistics = set(statistics)
    all_combinations = set([f"{q}.{s}" for q in all_quantities for s in all_statistics])
    existing_combinations = set([c for c in df.columns if "." in c])
    missing_combinations = all_combinations - existing_combinations

    if len(missing_combinations) > 0:
        df = df.with_columns((pl.lit(None).alias(c) for c in missing_combinations))
    return df


def get_timeseries(graph, preds_succs, engine, store_subgraphs=False, subgraph_storagepath=None):
    """Get timeseries data from the database for the cells in the graph."""
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="raincells", views=True)
    stat_view = metadata.tables["raincells.raincells_with_statistics"]

    dfs = []
    for midnode in preds_succs.keys():
        if len(preds_succs[midnode]["predecessors"]) == 0 and len(preds_succs[midnode]["successors"]) == 0:
            logging.warning(f"Node {midnode} has no predecessors or successors. Skipping this node.")
            continue

        nodes = [midnode] + preds_succs[midnode]["predecessors"] + preds_succs[midnode]["successors"]
        nodes = sorted(nodes)

        keys = []
        events = []
        for node in nodes:
            data = graph.nodes[node]
            # print(f"Node: {node}, Data: {data}")

            method = data.get("method", "unknown")
            timestamp = data.get("timestamp", "unknown")
            identifier = data.get("identifier", "unknown")
            keys.append((timestamp, identifier, method))
            events.append((timestamp, identifier, method, data.get("event", "unknown")))

        query = select(
            stat_view.c.timestamp,
            stat_view.c.identifier,
            stat_view.c.method,
            stat_view.c.quantity,
            stat_view.c.statistic,
            stat_view.c.value,
            stat_view.c.area,
        ).where(
            or_(
                and_(stat_view.c.timestamp == t, stat_view.c.identifier == i, stat_view.c.method == m)
                for t, i, m in keys
            )
        )

        with Session(engine) as session:
            # sql queryn tulostus arvojen kanssa
            # print(str(query.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
            df_data = pl.read_database(query=query, connection=session.bind)

        # if midnode == "2021-07-13T11:55:00_17":
        #     import ipdb

        #     ipdb.set_trace()

        # sort by timestamp, identifier, method
        df_data = df_data.sort(["timestamp", "identifier", "method"])
        midnode_timestamp = graph.nodes[midnode]["timestamp"]
        node_type = preds_succs[midnode]["type"]
        df_data = df_data.with_columns(
            level=((pl.col("timestamp") - midnode_timestamp).dt.total_seconds() / 60 / 5).cast(pl.Int32)
        )

        df_data = df_data.with_columns(
            pl.lit(midnode).alias("t0_node"),
            pl.lit(node_type).alias("type"),
        )

        graph_data_df = pl.from_records(events, schema=["timestamp", "identifier", "method", "event"], orient="row")

        df_data = df_data.join(graph_data_df, on=["timestamp", "identifier", "method"], how="left")
        dfs.append(df_data)

        if store_subgraphs:
            subgraph = graph.subgraph(nodes)

            subgraph_filename = (
                Path(subgraph_storagepath.format(timestamp=midnode_timestamp.strftime("%Y%m%d")))
                / f"subgraph_{midnode}.gml"
            )
            subgraph_filename.parent.mkdir(parents=True, exist_ok=True)
            nx.write_gml(subgraph, subgraph_filename, stringizer=stringizer)

    if len(dfs) == 0:
        # logging.error("No data found for the given nodes.")
        return None

    df = pl.concat(dfs, how="diagonal_relaxed")
    df = df.with_columns(pl.format("{}.{}", "quantity", "statistic").alias("on")).pivot(
        on="on",
        # index=["type", "t0_node", "timestamp", "level", "identifier", "method", "area", "event"],
        index=set(df.columns) - set(["quantity", "statistic", "value"]),
        values=["value"],
    )
    return df


def plot_split_merge_timeseries(
    df,
    title="Split and Merge Timeseries",
    variable="rate.sum",
    variable_label="Total rain rate",
    merge_title="Merged nodes",
    split_title="Split nodes",
    outpath=Path("timeseries_plot.pdf"),
    lineplot_kwargs={},
):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
        layout="constrained",
    )
    sns.lineplot(
        data=df[df["type"] == "merged"],
        x="level",
        y=variable,
        # hue="t0_node",
        ax=axs[0],
        legend=False,
        **lineplot_kwargs,
    )
    sns.lineplot(
        data=df[df["type"] == "split"],
        x="level",
        y=variable,
        # hue="t0_node",
        ax=axs[1],
        legend=False,
        **lineplot_kwargs,
    )
    axs[0].set_title(merge_title)
    axs[1].set_title(split_title)
    for ax in axs:
        ax.set_xlabel("Time step (5 min)")
        ax.set_ylabel(variable_label)
        # ax.legend(title="Node", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    del fig


def remove_splits_merges_from_trajectory(timeseries_df):
    """Remove extra splits and merges from the trajectory.

    TODO: does not work at all as intended, needs to be fixed.

    Parameters
    ----------
    timeseries_df: pl.DataFrame
        DataFrame with the trajectory data.
    """
    # Get list of original columns to use later for picking output columns
    orig_columns = timeseries_df.columns

    # Operate on a copy of the DataFrame
    df = timeseries_df.clone()

    # Get the number of expected cells in the past and future of the t0_node
    # based on the number of nodes at the first time step before/after the t0_node
    expected_num_cells_before_t0 = (
        df.with_columns(
            pl.col("timestamp").count().over("type", "t0_node", "level").alias("expected_num_cells_before_t0")
        )
        .select("type", "t0_node", "level", "expected_num_cells_before_t0")
        .sort("type", "t0_node", "level")
        .filter(pl.col("level") == -1)
        .unique()
    )
    expected_num_cells_after_t0 = (
        df.with_columns(
            pl.col("timestamp").count().over("type", "t0_node", "level").alias("expected_num_cells_after_t0")
        )
        .select("type", "t0_node", "level", "expected_num_cells_after_t0")
        .sort("type", "t0_node", "level")
        .filter(pl.col("level") == 1)
        .unique()
    )
    # Combine the expected number of cells before and after t0 to one df
    expected_num_cells = expected_num_cells_before_t0.join(
        expected_num_cells_after_t0,
        on=["type", "t0_node"],
        how="left",
    ).drop("level", "level_right")

    df = (
        df.join(
            expected_num_cells,
            on=["type", "t0_node"],
            how="full",
        )
        # In one column the expected number of cells at each level
        .with_columns(
            pl.when(pl.col("level") < 0)
            .then(pl.col("expected_num_cells_before_t0"))
            .otherwise(pl.col("expected_num_cells_after_t0"))
            .alias("expected_num_cells"),
        )
        .with_columns(
            # Check if the number of cells at each level is valid
            valid_level=(
                ((pl.col("level") < 0) & (pl.col("num_cells_at_level") == pl.col("expected_num_cells")))
                | (pl.col("level") == 0)
                | ((pl.col("level") > 0) & (pl.col("num_cells_at_level") == pl.col("expected_num_cells")))
            ),
            # Group the levels by type, t0_node and level (before and after t0_node)
            level_group=pl.struct(
                "type",
                "t0_node",
                pl.when(pl.col("level") != 0).then(pl.col("level").sign()).otherwise(0).alias("level_group"),
            ),
        )
        .sort("type", "t0_node", "level")
        .drop(
            "expected_num_cells_before_t0",
            "expected_num_cells_after_t0",
            "type_right",
            "t0_node_right",
            # "expected_num_cells_before_t0_right",
            # "expected_num_cells_after_t0_right",
        )
    )

    valid_levels = (
        df.select(
            "type",
            "t0_node",
            "level",
            "timestamp",
            "valid_level",
            "num_cells_at_level",
            "expected_num_cells",
            "level_group",
        )
        .unique(subset=["type", "t0_node", "level"])
        .sort("t0_node", "type", pl.col("level").abs())
        .with_columns(
            # Calculate cumulative sum of valid levels before and after t0_node
            # to determine if the level and subsequent levels are valid
            (pl.col("valid_level") * pl.col("level") < 0)
            .cum_sum()
            .over("level_group")
            .alias("valid_level_cumsum_before"),
            (pl.col("valid_level") * pl.col("level") > 0)
            .cum_sum()
            .over("level_group")
            .alias("valid_level_cumsum_after"),
        )
        # If all levels before this one have been valid and this one is valid also (i.e. cumsum of valid levels equals level),
        # then this level is valid
        .with_columns(valid_level_cumsum=pl.col("valid_level_cumsum_before") + pl.col("valid_level_cumsum_after"))
        # Remove the levels that are not valid
        .filter(pl.col("valid_level_cumsum") == pl.col("level").abs())
    )

    # Join the valid levels with the original DataFrame
    df_valid = df.join(valid_levels, how="right", on=["type", "t0_node", "level"], suffix="_y", coalesce=True)
    # Remove the columns that are not needed
    df_valid = df_valid.select(  # .filter(pl.col("t0_node") == "2021-07-13T00:45:00_24")
        *orig_columns,
        # "type",
        # "t0_node",
        # "level",
        # "timestamp",
        "valid_level",
        # "num_cells_at_level",
        # "expected_num_cells",
        "valid_level_cumsum",
    ).sort("t0_node", "type", "level")

    return df_valid
