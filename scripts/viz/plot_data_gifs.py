"""Plot input data gifs in separate figures.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""

import argparse
import os
from copy import copy
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyart
from matplotlib.collections import LineCollection
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
import distinctipy

from utils.config_utils import load_config
from utils.data_utils import load_data
from utils.plot_utils import plot_array, convert_colorspace

import database.cell_queries as cell_queries

pyart.load_config(os.environ.get("PYART_CONFIG"))

bad_times_file = Path("bad_times_input_data_plotting.csv")


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

    # Set up figure
    ncols = conf.n_images
    m, n = conf.im_size
    input_data_conf = {k: v for k, v in conf.input_data.items() if k in conf.plot_variables}
    nrows = len(input_data_conf.keys())

    if conf.plot_map:
        # Borders
        border = gpd.read_file(conf.map_params.border_shapefile)
        border_proj = border.to_crs(conf.map_params.proj)

        segments = [np.array(linestring.coords)[:, :2] for linestring in border_proj["geometry"]]
        border_collection = LineCollection(segments, zorder=0, **conf.map_params.border_plot_kwargs)

        # Radar locations
        if conf.map_params.radar_shapefile is not None:
            radar_locations = gpd.read_file(conf.map_params.radar_shapefile)
            radar_locations_proj = radar_locations.to_crs(conf.map_params.proj)
            xy = radar_locations_proj["geometry"].map(lambda point: point.xy)
            radar_locations_proj = list(zip(*xy))
        else:
            radar_locations_proj = None

        # Radar limits
        radar_limits = []
        if conf.map_params.radar_limits is not None:
            for limit_name, limit_conf in conf.map_params.radar_limits.items():
                if limit_conf.shapefile is not None:
                    limit = gpd.read_file(limit_conf.shapefile)
                    limit = limit.set_crs("EPSG:21781", allow_override=True).to_crs(conf.map_params.proj)
                    radar_limits.append((limit, limit_conf.plot_kwargs))

    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(conf.col_width + 1, conf.row_height),
        sharex="col",
        sharey="row",
        squeeze=True,
        constrained_layout=True,
    )

    out_file = "{name}_{time:%Y%m%d%H%M}.{ext}"

    get_times = pd.date_range(start=date, periods=conf.n_images, freq=conf.freq).to_pydatetime().tolist()

    missing_times_vars = []
    try:
        dataset = load_data(
            input_data_conf,
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

    cells = None
    cell_methods = [""]
    if conf.plot_cells:
        # Load cell tracks
        print(f"Loading cell tracks")
        trackpath = Path(conf.cell_tracks["storage_path"])
        fileglob = trackpath / conf.cell_tracks["simple_tracks_fileglob"]

        if "csv" in conf.cell_tracks["simple_tracks_fileglob"]:
            df_tracks = pl.read_csv(fileglob, try_parse_dates=True)
        elif "parquet" in conf.cell_tracks["simple_tracks_fileglob"]:
            df_tracks = pl.read_parquet(fileglob)
        else:
            raise ValueError(f"Unknown file type {conf.cell_tracks['simple_tracks_fileglob']}")

        df_tracks = df_tracks.with_columns(
            pl.col("t0_node").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("start_time"),
            pl.col("t0_node").str.slice(20, 21).cast(pl.Int32).alias("start_id"),
        )

        try:
            df_tracks = df_tracks.with_columns(
                pl.col("start_time").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        except pl.exceptions.SchemaError:
            pass

        # Drop tracks that are only one time step long
        df_tracks = df_tracks.filter(pl.col("timestamp").n_unique().over("start_time", "start_id", "method") > 1)

        # Get tracks that exist at the time we are plotting
        df_tracks = df_tracks.filter((pl.col("timestamp") >= get_times[0]) & (pl.col("timestamp") <= get_times[-1]))

        cell_methods = list(conf.cell_options.keys())

        celltracks = {}
        track_cmaps = {}
        num_cell_tracks = {}
        all_cells = []
        track_types = set()
        for i, cell_method in enumerate(cell_methods):
            if cell_method == "no_tracks":
                continue

            plotted_track_types = list(conf.cell_options[cell_method].keys())
            track_types.update(plotted_track_types)

            cells = cell_queries.load_stormcells_between_times(
                get_times[0],
                get_times[-1],
                conf.cell_database_config,
                cell_method=cell_method,
            )
            all_cells.append(cells)

            num_cell_tracks[cell_method] = (
                df_tracks.filter(pl.col("method") == cell_method)
                .filter(pl.col("type").is_in(plotted_track_types))
                .select(pl.struct("start_time", "start_id").n_unique())
                .item()
            )

            track_cmaps[cell_method] = distinctipy.get_colors(
                num_cell_tracks[cell_method],
                pastel_factor=min(i * 0.2, 1.0),
                colorblind_type="Deuteranomaly",
                n_attempts=100,
            )

            celltracks[cell_method] = (
                df_tracks.filter(pl.col("method") == cell_method)
                .filter(pl.col("type").is_in(plotted_track_types))
                .select(pl.col("start_time", "start_id"))
                .unique()
                .to_pandas()
            )
        cells = pd.concat(all_cells)
        df_tracks = df_tracks.filter(pl.col("type").is_in(track_types)).to_pandas()
        df_cells = cells.merge(df_tracks, on=["timestamp", "identifier", "method"], how="left")

    # Plot data
    for cell_method in tqdm(cell_methods):
        tqdm.write(f"Plotting {cell_method} cells")
        outdir = Path(date.strftime(conf.outdir.format(cell_method=cell_method)))
        outdir.mkdir(parents=True, exist_ok=True)

        for j, name in tqdm(enumerate(input_data_conf.keys())):
            var_name = input_data_conf[name].variable

            files = []
            for i, (time, arr) in tqdm(enumerate(dataset.groupby("time", squeeze=False))):
                tqdm.write(f"Plotting {name} at {time}")
                im = arr[var_name].to_numpy().squeeze()

                nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
                zero_mask = np.isclose(im, 0)

                im[zero_mask] = np.nan
                im[nan_mask] = np.nan

                cbar = None
                if not conf.plot_empty_data or "no_tracks" in cell_method:
                    # Plot input data
                    cbar = plot_array(
                        axs,
                        im.copy(),
                        x=dataset.x.values,
                        y=dataset.y.values,
                        qty=input_data_conf[name].cmap_qty,
                        colorbar=conf.plot_cbar,
                        zorder=1,
                    )
                # Remove cbar title
                if cbar is not None:
                    cbar.set_label("")
                axs.set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}", va="bottom")

                if conf.plot_map:
                    axs.add_collection(copy(border_collection))

                    if radar_locations_proj is not None:
                        axs.scatter(
                            *radar_locations_proj,
                            zorder=5,
                            **conf.map_params.radar_plot_kwargs,
                        )
                    for limit, plot_kwargs in radar_limits:
                        axs.plot(*limit.geometry.values[0].exterior.xy, zorder=5, **plot_kwargs)

                if conf.plot_cells:
                    cells_for_this_time = df_cells[df_cells.timestamp == time]

                    # Get cells for this method
                    if "no_tracks" in cell_method:
                        cells_to_plot = cells_for_this_time
                    else:
                        cells_to_plot = cells_for_this_time[cells_for_this_time["method"] == cell_method]

                    # Get cells that are part of track (ie, start_time and start_id not null)
                    cells_in_tracks = cells_to_plot[
                        (~cells_to_plot["start_time"].isnull()) & (~cells_to_plot["start_id"].isnull())
                    ]
                    cells_not_in_tracks = cells_to_plot[
                        (cells_to_plot["start_time"].isnull()) | (cells_to_plot["start_id"].isnull())
                    ]

                    track_plot_kwargs = copy(conf.default_cell_plot_kwargs)
                    if conf.cell_options[cell_method] is not None:
                        args_without_edgecolor = copy(conf.default_cell_plot_kwargs)
                        args_without_edgecolor.pop("edgecolor", None)
                        track_plot_kwargs.update(args_without_edgecolor)

                    cells_not_in_tracks.plot(
                        ax=axs,
                        zorder=20,
                        autolim=False,
                        **track_plot_kwargs,
                        # **conf.cell_options[cell_method].plot_kwargs,
                    )

                    if "no_tracks" not in cell_method:
                        # track_plot_kwargs = copy(conf.default_cell_plot_kwargs)
                        # track_plot_kwargs.update(conf.cell_options[cell_method].plot_kwargs)

                        for it, track in cells_in_tracks.iterrows():
                            ttype = track["type"]
                            if conf.cell_options[cell_method][ttype].plot_kwargs.get("edgecolor") == "auto":
                                cmap_idx = celltracks[cell_method][
                                    (celltracks[cell_method].start_time == track.start_time)
                                    & (celltracks[cell_method].start_id == track.start_id)
                                ].index.item()
                                track_plot_kwargs["edgecolor"] = track_cmaps[cell_method][cmap_idx]
                            else:
                                track_plot_kwargs["edgecolor"] = conf.cell_options[cell_method][ttype].plot_kwargs.get(
                                    "edgecolor", conf.default_cell_plot_kwargs.get("edgecolor", "black")
                                )

                            # If cell is start_time and start_id, plot it with dashed line
                            if track.identifier == track.start_id and track.timestamp == track.start_time:
                                track_plot_kwargs["linestyle"] = "--"
                            else:
                                track_plot_kwargs["linestyle"] = "-"

                            gpd.GeoSeries(track.geometry).plot(
                                ax=axs,
                                zorder=25,
                                # edgecolor=edgecolor,
                                **track_plot_kwargs,
                            )

                axs.set_xticks(
                    np.arange(
                        dataset.x.values.min(),
                        dataset.x.values.max(),
                        conf.tick_spacing * 1e3,
                    )
                )
                axs.set_yticks(
                    np.arange(
                        dataset.y.values.min(),
                        dataset.y.values.max(),
                        conf.tick_spacing * 1e3,
                    )
                )
                axs.set_aspect(1)

                if conf.zoom_bbox is not None:
                    im_width = dataset.x.values.max() - dataset.x.values.min()
                    im_height = dataset.y.values.max() - dataset.y.values.min()

                    axs.set_xlim(
                        (
                            dataset.x.values.min() + im_width * conf.zoom_bbox[0],
                            dataset.x.values.min() + im_width * conf.zoom_bbox[1],
                        )
                    )
                    axs.set_ylim(
                        (
                            dataset.y.values.min() + im_height * conf.zoom_bbox[2],
                            dataset.y.values.min() + im_height * conf.zoom_bbox[3],
                        )
                    )

                else:
                    axs.set_xlim((dataset.x.values.min(), dataset.x.values.max()))
                    axs.set_ylim((dataset.y.values.min(), dataset.y.values.max()))

                if conf.get("plot_bbox", None) is not None:
                    im_width = dataset.x.values.max() - dataset.x.values.min()
                    im_height = dataset.y.values.max() - dataset.y.values.min()
                    bbox_x = [
                        dataset.x.values.min() + im_width * conf.plot_bbox[0],
                        dataset.x.values.min() + im_width * conf.plot_bbox[1],
                    ]
                    bbox_y = [
                        dataset.y.values.min() + im_height * conf.plot_bbox[2],
                        dataset.y.values.min() + im_height * conf.plot_bbox[3],
                    ]

                    axs.plot(
                        [bbox_x[0], bbox_x[-1], bbox_x[-1], bbox_x[0], bbox_x[0]],
                        [bbox_y[0], bbox_y[0], bbox_y[-1], bbox_y[-1], bbox_y[0]],
                        **conf.get("bbox_plot_kwargs", dict(color="red", lw=1, ls="solid", alpha=1.0)),
                        zorder=15,
                    )

                axs.xaxis.set_major_formatter(plt.NullFormatter())
                axs.yaxis.set_major_formatter(plt.NullFormatter())

                axs.grid(lw=0.8, color="tab:gray", ls=":", zorder=11)

                for tick in axs.xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
                for tick in axs.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

                for spine in ["top", "right"]:
                    axs.spines[spine].set_visible(True)

                if conf.plot_qty_title and (not conf.plot_empty_data):
                    # axs.set_ylabel(input_data_conf[name]["title"])
                    axs.text(
                        0.02,
                        0.978,
                        f"{input_data_conf[name]['title']}",
                        ha="left",
                        va="top",
                        color=plt.rcParams.get("axes.labelcolor"),
                        fontsize=plt.rcParams.get("axes.labelsize"),
                        fontweight=plt.rcParams.get("axes.labelweight"),
                        transform=axs.transAxes,
                        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black", boxstyle="square,pad=0.5"),
                        zorder=30,
                    )

                # fontprops = mpl.font_manager.FontProperties(size="medium")
                # scalebar = AnchoredSizeBar(
                #     axs.transData,
                #     100e3,
                #     '100km',
                #     'upper right',
                #     pad=0.7,
                #     color='k',
                #     sep=5,
                #     frameon=False,
                #     size_vertical=2,
                #     label_top=True,
                #     fontproperties=fontprops
                # )

                # axs.add_artist(scalebar)

                imfile = outdir / out_file.format(name=name, time=pd.Timestamp(time), ext=conf.get("ext", "png"))
                files.append(imfile)
                fig.savefig(
                    imfile,
                    # bbox_inches="tight",
                    dpi=conf.dpi,
                )
                if conf.get("ext", "png") == "png":
                    convert_colorspace(imfile)
                if cbar is not None:
                    cbar.remove()
                axs.clear()

            if not conf.make_gif:
                continue

            frames = np.stack(
                [iio.imread(filename) for filename in sorted(files)],
                axis=0,
            )

            iio.imwrite(
                outdir / f"{name}_{date:%Y%m%d%H%M}.gif",
                frames,
                duration=conf.duration_per_frame * 1000,
                loop=0,
            )
