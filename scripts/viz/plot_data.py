"""Plot input data.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import pyart
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import xarray as xr
import geopandas as gpd
from copy import copy
from matplotlib.collections import LineCollection
from tqdm import tqdm

from pathlib import Path

from utils.config_utils import load_config
from utils.plot_utils import plot_array
from database.cell_queries import load_stormcells_between_times

pyart.load_config(os.environ.get("PYART_CONFIG"))

from utils.data_utils import (
    load_data,
)

bad_times_file = Path("bad_times_input_data_plotting.csv")

BBOX_LW = 1.5
BBOX_COL = "tab:red"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("configpath", type=str, help="Configuration file path")
    argparser.add_argument("date", type=str, help="date to be plotted (YYYYmmddHHMM)")
    args = argparser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d%H%M")
    sample = date.strftime("%Y-%m-%d %H:%M:%S")

    confpath = Path(args.configpath)
    conf = load_config(confpath)
    plt.style.use(conf.stylefile)

    outdir = Path(date.strftime(conf.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Set up figure
    ncols = conf.n_images
    m, n = conf.im_size
    nrows = len(conf.input_data.keys())

    if conf.plot_map:
        # Borders
        border = gpd.read_file(conf.map_params.border_shapefile)
        border_proj = border.to_crs(conf.map_params.proj)

        segments = [np.array(linestring.coords)[:, :2] for linestring in border_proj["geometry"]]
        border_collection = LineCollection(segments, zorder=10, **conf.map_params.border_plot_kwargs)

        # Radar locations
        if conf.map_params.radar_shapefile is not None:
            radar_locations = gpd.read_file(conf.map_params.radar_shapefile)
            radar_locations_proj = radar_locations.to_crs(conf.map_params.proj)
            xy = radar_locations_proj["geometry"].map(lambda point: point.xy)
            radar_locations_proj = list(zip(*xy))
        else:
            radar_locations_proj = None

    if conf.figsize is not None:
        figsize = conf.figsize
    else:
        figsize = (ncols * conf.col_width + 1, nrows * conf.row_height)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex="col",
        sharey="row",
        squeeze=False,
        height_ratios=[*[1] * nrows],
        layout="constrained",
    )

    get_times = pd.date_range(start=date, periods=conf.n_images, freq=conf.freq).to_pydatetime().tolist()

    missing_times_vars = []
    try:
        dataset = load_data(
            conf.input_data,
            date,
            get_times,
            len(get_times),
            None,
        )
    except (FileNotFoundError, ValueError) as e:
        with open(bad_times_file, "a") as f:
            f.write(f"{pd.Timestamp(date)},{e},\n")
        raise FileNotFoundError(f"Loading data failed for {date}") from e

    # Check if we have any missing data
    for var, da in dataset.data_vars.items():
        for tt in dataset.time.values:
            if ("input_files" in da.attrs.keys()) and np.all(np.isnan(da.sel(time=tt).values)):
                missing_times_vars.append((tt, var))

    if len(missing_times_vars):
        problem_times = []
        min_times = sorted(dataset.time.values)[: conf.n_images]
        for tt, var in missing_times_vars:
            if tt in min_times:
                with open(bad_times_file, "a") as f:
                    f.write(f"{pd.Timestamp(tt)},{var},\n")
                problem_times.append((pd.Timestamp(tt), var))
        raise FileNotFoundError(f"Files missing for {problem_times}")

    bbox_xx = dataset.x.values
    bbox_yy = dataset.y.values
    if conf.get("plot_bbox") is not None:
        bbox_yy = bbox_yy[conf.plot_bbox[0] : conf.plot_bbox[1]]
        bbox_xx = bbox_xx[conf.plot_bbox[2] : conf.plot_bbox[3]]

    cells = None
    if conf.plot_cells:
        cells = load_stormcells_between_times(
            get_times[0],
            get_times[-1],
            conf.cell_database_config,
        )
        cell_methods = list(conf.cell_options.keys())

    # Plot data
    row = 0
    for j, name in tqdm(enumerate(conf.input_data.keys())):
        row = j
        var_name = conf.input_data[name].variable

        for i, (time, arr) in enumerate(dataset.groupby("time")):
            im = arr[var_name].to_numpy().squeeze()

            nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
            zero_mask = np.isclose(im, 0)

            im[zero_mask] = np.nan
            im[nan_mask] = np.nan

            # Plot input data
            plot_array(
                axes[row, i],
                im.copy(),
                x=dataset.x.values,
                y=dataset.y.values,
                qty=conf.input_data[name].cmap_qty,
                colorbar=(i == (ncols - 1)),
            )
            axes[row, i].set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}")

            if conf.plot_map:
                axes[row, i].add_collection(copy(border_collection))

                if radar_locations_proj is not None:
                    axes[row, i].scatter(
                        *radar_locations_proj,
                        zorder=10,
                        **conf.map_params.radar_plot_kwargs,
                    )

            if conf.plot_bbox is not None:
                # Plot a box around the nowcast area
                axes[row, i].plot(
                    bbox_xx[[0, -1, -1, 0, 0]],
                    bbox_yy[[0, 0, -1, -1, 0]],
                    color=BBOX_COL,
                    lw=BBOX_LW,
                    zorder=15,
                )

            if conf.plot_cells:
                cells_for_this_time = cells[cells.timestamp == time]
                for method in cell_methods:
                    cells_to_plot = cells_for_this_time[cells_for_this_time["method"] == method]
                    cells_to_plot.plot(
                        ax=axes[row, i],
                        zorder=20,
                        autolim=False,
                        **conf.cell_options[method].plot_kwargs,
                    )

        axes[row, 0].set_ylabel(conf.input_data[name]["title"])

    for ax in axes.flat:
        ax.set_xticks(np.arange(dataset.x.values.min(), dataset.x.values.max(), conf.tick_spacing * 1e3))
        ax.set_yticks(np.arange(dataset.y.values.min(), dataset.y.values.max(), conf.tick_spacing * 1e3))
        ax.set_aspect(1)

        ax.set_xlim((dataset.x.values.min(), dataset.x.values.max()))
        ax.set_ylim((dataset.y.values.min(), dataset.y.values.max()))

        ax.grid(lw=0.5, color="tab:gray", ls=":", zorder=11)

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

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(True)

    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.savefig(
        outdir / date.strftime(conf.filename),
        bbox_inches="tight",
        dpi=conf.dpi,
    )
