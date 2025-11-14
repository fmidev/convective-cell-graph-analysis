"""Functions for plotting data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import polars as pl
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors, cm, gridspec, ticker, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import colormaps as cmaps  # noqa

from . import quantity_definitions
from . import plot_utils

lineplot_kwargs = {
    "errorbar": ("ci", 95),
}
count_linekwargs = {
    "ls": "--",
    "lw": 1,
    "color": "black",
    "label": "Subgraph count",
}


def plot_trajectory_development_figure(
    trajectories: pl.DataFrame,
    t0_types: list[str],
    variables: list[str],
    variable_labels: list[str],
    row_title_funcs: list[callable],
    colors: list[str],
    outfilename: str,
    lineplot_kwargs=lineplot_kwargs,
    count_linekwargs=count_linekwargs,
    include_control=True,
    ylim=(-50, 250),
    xlim=(-6, 6),
    count_ylim=(0, 12),
    count_ytick_multiples=(2, 1),
    xtick_multiples=(2, 1),
    ytick_multiples=(50, 25),
    extensions=["png", "pdf"],
    control_suffix="_control",
    count_multiply=1000,
    outpath=".",
    savefig_kwargs=dict(dpi=300, transparent=False),
    grid_kwargs=dict(which="both", linestyle="--", linewidth=0.5, alpha=0.7),
    zero_line_kwargs=dict(color="black", linestyle="-", lw=1.5, zorder=5),
    legend_locs=defaultdict(lambda: "best"),
    legend_ncols="auto",
    row_height=3,
    col_width=8,
    linestyles=None,
    plot_counts=True,
    figure_direction="vertical",
    return_fig=False,
):
    if include_control:
        t0_types_ = t0_types + [f"{t}{control_suffix}" for t in t0_types]
        control_types = [f"{t}{control_suffix}" for t in t0_types]
    else:
        t0_types_ = t0_types

    if figure_direction == "vertical":
        ncols = 2 if include_control else 1
        nrows = len(t0_types)
        flatten_order = "C"
    else:
        nrows = 2 if include_control else 1
        ncols = len(t0_types)
        flatten_order = "F"
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * col_width, row_height * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    df = trajectories.filter(pl.col("type").is_in(t0_types_)).to_pandas()

    if flatten_order == "F":
        ax = axs[0, :]
    else:
        ax = axs[:, 0]

    _, twin_axs = plot_trajectory_development(
        trajectories=df,
        t0_types=t0_types,
        quantities=variables,
        labels=variable_labels,
        colors=colors,
        axs=ax,
        lineplot_kwargs=lineplot_kwargs,
        count_linekwargs=count_linekwargs,
        count_multiply=count_multiply,
        plot_counts=plot_counts,
        linestyles=linestyles,
        flatten_order=flatten_order,
    )
    if include_control:
        _, twin_axs_ = plot_trajectory_development(
            trajectories=df,
            t0_types=control_types,
            quantities=variables,
            labels=variable_labels,
            colors=colors,
            axs=ax,
            lineplot_kwargs=lineplot_kwargs,
            count_linekwargs=count_linekwargs,
            count_multiply=count_multiply,
            plot_counts=plot_counts,
            linestyles=linestyles,
            flatten_order=flatten_order,
        )
        twin_axs = twin_axs + twin_axs_

        for i, control_group in enumerate(control_types):
            axs[i, 1].set_title(f"{row_title_funcs[i](control_group)}")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    count_handles, count_labels = twin_axs[0].get_legend_handles_labels()

    for i, t0_type in enumerate(t0_types):
        if flatten_order == "F":
            axs[0, i].set_title(row_title_funcs[i](t0_types[i]))
        else:
            axs[i, 0].set_title(row_title_funcs[i](t0_types[i]))

    if legend_ncols == "auto":
        legend_ncols = len(variables) + 1

    for i, ax in enumerate(axs.flatten(order=flatten_order)):
        ax.set_ylabel("Relative change from $t_0$ [%]")
        ax.set_xlabel("Timestep [5min]")
        ax.set_ylim(ylim)
        ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(ytick_multiples[0]))
        ax.axes.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_multiples[1]))
        ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(xtick_multiples[0]))
        ax.axes.xaxis.set_minor_locator(ticker.MultipleLocator(xtick_multiples[1]))
        ax.grid(**grid_kwargs)
        ax.set_xlim(xlim)
        ax.axhline(y=0, **zero_line_kwargs)
        # remove the legend from the axes
        ax.get_legend().remove()
        ax.legend(
            handles=handles + count_handles,
            labels=labels + count_labels,
            loc=legend_locs[i],
            frameon=True,
            ncol=legend_ncols,
            facecolor="white",
            edgecolor="white",
        )

    for ax in twin_axs:
        ax.set_ylim(count_ylim)
        ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(count_ytick_multiples[0]))
        ax.axes.yaxis.set_minor_locator(ticker.MultipleLocator(count_ytick_multiples[1]))

    outname = outfilename
    plot_utils.save_figs(
        fig=fig,
        delete_fig=(not return_fig),
        outpath=outpath,
        name=outname,
        extensions=extensions,
        savefig_kwargs=savefig_kwargs,
    )
    if return_fig:
        return fig, axs, twin_axs


def plot_seasonal_cycle(
    dataframes: list[pl.DataFrame],
    labels: list[str],
    colors: list[str],
    axs: list[plt.Axes],
    hue_col="name",
    shrink=0.6,
    hist_alpha=0.2,
    hist_lw=1.5,
    diurnal_cycle_hist_alpha=0.2,
    legend_fontsize="small",
):
    """Plot seasonal and diurnal cycle of cells.


    Parameters
    ----------
    dataframes : list[pl.DataFrame]
        List of polars DataFrames containing the data to plot. Each dataframe should have columns 'month' and 'local_hour'.
    labels : list[str]
        List of labels for each dataframe, used for the legend.
    colors : list[str]
        List of colors for each label, used for the plot.
    axs : list[plt.Axes]
        List of matplotlib Axes objects where the plots will be drawn. Needs to have at least two axes for seasonal and diurnal cycles.
    hue_col : str, optional
        Column name to use for hue in the plot, by default "name".
    shrink : float, optional
        Shrink factor for the bars in seasonal cycle plot, by default 0.6.
    hist_alpha : float, optional
        Alpha value for the histogram fills, by default 0.2.
    hist_lw : float, optional
        Line width for the histogram edges, by default 1.5.

    Returns
    -------
    axs : list[plt.Axes]
        List of matplotlib Axes objects with the plots.
    """
    col = 0

    # Plot seasonal cycle of cells
    # Join dataframes into a single dataframe
    dfs = []
    for df, label in zip(dataframes, labels):
        df_ = df.select("month").with_columns(pl.lit(label).alias(hue_col)).to_pandas()
        dfs.append(df_)

    df = pd.concat(dfs)

    g1 = sns.histplot(
        data=df,
        x="month",
        hue=hue_col,
        hue_order=labels,
        multiple="dodge",
        ax=axs[col],
        stat="percent",
        palette={lab: c for lab, c in zip(labels, colors)},
        # color=[colors["all"], colors["t0"]],
        # facecolor=colors["all"],
        # edgecolor={t0_type: colors["t0"], "All cells": colors["all"]},
        element="bars",
        linewidth=hist_lw,
        alpha=hist_alpha + 0.1,
        # edgecolor=None,
        # bins=5,
        binwidth=1.0,
        binrange=(5, 9),
        discrete=True,
        legend=False,
        rasterized=True,
        shrink=shrink,
        common_norm=False,
    )

    # Set bar edgecolor to the same as facecolor
    for patch in g1.patches:
        facecolor = list(patch.get_facecolor())
        facecolor[-1] = 1
        patch.set_edgecolor(facecolor)
        patch.set_linewidth(hist_lw)

    axs[col].set_title(f"Monthly distributions")
    axs[col].set_xlabel("Month")
    axs[col].set_ylabel("Proportion [%]")
    axs[col].set_xticks([5, 6, 7, 8, 9])
    axs[col].set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
    # axs[col].xaxis.set_major_locator(ticker.MultipleLocator(3))
    # axs[col].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[col].grid(axis="y", which="major")
    axs[col].grid(axis="y", which="minor")
    axs[col].set_xlim(4.3, 9.8)

    # Diurnal cycle
    col = 1

    for df, label, color in zip(dataframes, labels, colors):
        sns.histplot(
            data=df.to_pandas(),
            x="local_hour",
            ax=axs[col],
            stat="percent",
            color=color,
            edgecolor=color,
            element="step",
            linewidth=hist_lw,
            alpha=diurnal_cycle_hist_alpha,
            bins=24,
            binrange=(0, 24),
            discrete=True,
            label=label,
            legend=True,
            rasterized=True,
            # multiple="dodge",
        )

    # Create legends
    legend_handles, legend_labels = axs[col].get_legend_handles_labels()

    axs[0].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        fontsize=legend_fontsize,
        framealpha=0.8,
        edgecolor="black",
        facecolor="white",
    )

    axs[1].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        fontsize=legend_fontsize,
        framealpha=0.8,
        edgecolor="black",
        facecolor="white",
    )

    axs[col].set_title(f"Diurnal cycle")
    axs[col].set_xlabel("Local hour [UTC + 2]")
    axs[col].set_ylabel("Proportion [%]")
    axs[col].xaxis.set_major_locator(ticker.MultipleLocator(3))
    axs[col].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[col].grid(axis="both", which="major")
    # axs[col].grid(axis="both", which="minor")
    axs[col].set_xlim(-0.5, 23.5)

    return axs


def plot_trajectory_development(
    trajectories: pd.DataFrame,
    t0_types: list[str],
    quantities: list[str],
    labels: list[str],
    colors: list[str],
    axs: plt.Axes = None,
    count_linekwargs: dict = dict(color="tab:green", linestyle="--", label="Subgraph count"),
    lineplot_kwargs: dict = {},
    count_multiply: float = 1000.0,
    plot_counts: bool = True,
    linestyles=None,
    flatten_order="C",
):
    """Plot the trajectory development for each t0 type.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame containing the trajectory data with columns 'type', 'level', and the quantities to plot.
    t0_types : list[str]
        List of t0 types to plot, e.g., ["t0", "all"].
    quantities : list[str]
        List of quantities to plot. The quantities should be columns in the trajectories DataFrame.
    labels : list[str]
        List of labels for each quantity, used for the legend.
    colors : list[str]
        List of colors for each quantity, used for the plot.
    axs : plt.Axes, optional
        Matplotlib Axes object to plot on.
    count_linekwargs : dict, optional
        Keyword arguments for the count line plot, by default dict(color="tab:green", linestyle="--", label="Subgraph count").
    lineplot_kwargs : dict, optional
        Keyword arguments for the line plot in sns.lineplot, by default an empty dictionary.
    count_multiply : float, optional
        Factor to multiply the trajectory counts by for better visibility, by default 1000.0
    plot_counts : bool, optional
        Whether to plot the trajectory counts on a secondary y-axis, by default True.

    Returns
    -------
    axs : list[plt.Axes]
        List of matplotlib Axes objects with the plots.
    twin_axes : list[plt.Axes]
        List of secondary y-axes for the trajectory counts.
    """
    max_counts = []
    twin_axes = []

    if linestyles is None:
        linestyles = ["solid"] * len(quantities)

    for j, t0_type in enumerate(t0_types):

        # Plot the three variables
        for var, label, color, ls in zip(quantities, labels, colors, linestyles):
            lineplot_kwargs_ = lineplot_kwargs.copy()
            lineplot_kwargs_["linestyle"] = ls
            sns.lineplot(
                data=trajectories[trajectories["type"] == t0_type],
                x="level",
                y=var,
                hue=None,
                ax=axs[j],
                legend=True,
                label=label,
                color=color,
                zorder=10,
                **lineplot_kwargs_,
            )

        axs[j].set_title(t0_type)
        axs[j].set_xlabel("Timestep [5min]")
        axs[j].grid(True)
        axs[j].set_ylim(-40, 250)

        if not plot_counts:
            continue
        # Add second y-axis
        ax2 = axs[j].twinx()

        ax2.set_zorder(axs[j].get_zorder() - 1)
        axs[j].patch.set_visible(False)

        # ax2.set_yscale("log")
        # ax2.axes.yaxis.set_major_formatter(ticker.EngFormatter(unit="", places=0))
        twin_axes.append(ax2)

        counts_ = trajectories.groupby(["type", "level"]).count()["t0_node"] / count_multiply
        counts_ = counts_.reset_index()

        max_counts.append(counts_[counts_["type"] == t0_type].max()["t0_node"])
        sns.lineplot(
            data=counts_[counts_["type"] == t0_type],
            x="level",
            y="t0_node",
            ax=ax2,
            legend=False,
            zorder=5,
            **count_linekwargs,
            **lineplot_kwargs,
        )
        ax2.set_ylabel(f"Subgraph count [{count_multiply:.0f}]")
        ax2.label_outer()
        ax2.set_ylim(0, None)

    # Share the y-axis with the maximum count
    if plot_counts:
        try:
            max_count_index = np.nanargmax(max_counts)
        except ValueError:
            max_count_index = 0
        for i, ax2 in enumerate(twin_axes):
            if i != max_count_index:
                ax2.sharey(twin_axes[max_count_index])

    return axs, twin_axes


def plot_trajectory_development_multigroup(
    trajectories: pd.DataFrame,
    hue_col: str,
    hue_order: list[str],
    hue_labels: list[str],
    quantity: str,
    cmap="viridis",
    norm=None,
    axs: plt.Axes = None,
    estimator="mean",
    lineplot_kwargs: dict = {},
    add_cbar=True,
    fig=None,
    cbar_kwargs: dict = dict(orientation="vertical", label="Proportion [%]"),
    cax_kwargs: dict = dict(width="5%", height="100%", loc="center left", bbox_to_anchor=(1.05, 0, 1, 1), borderpad=0),
):
    """Plot the trajectory development for each group defined by hue_col.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame containing the trajectory data with columns 'type', 'level', and the quantities to plot.
    hue_col : str
        Column name to use for hue in the plot, e.g., "name" or "type".
    hue_order : list[str]
        List of hue categories to order the hue in the plot.
    hue_labels : list[str]
        List of labels for each hue category, used for the legend.
    quantity : str
        The quantity to plot on the y-axis. This should be a column in the trajectories DataFrame.
    cmap : str or colormap, optional
        Colormap to use for the hue, by default "viridis". Can be a string name or a colormap object.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance to scale the colormap, by default None. If None, a default normalization will be used.
    axs : plt.Axes, optional
        Matplotlib Axes object to plot on.
    estimator : str, optional
        Estimator to use for the line plot, e.g., "mean", "median".
    count_linekwargs : dict, optional
        Keyword arguments for the count line plot, by default dict(color="tab:green", linestyle="--", label="Subgraph count").
    lineplot_kwargs : dict, optional
        Keyword arguments for the line plot in sns.lineplot, by default an empty dictionary.
    add_cbar : bool, optional
        Whether to add a colorbar to the plot, by default True.
    fig : plt.Figure, optional
        Matplotlib Figure object to use for the colorbar.
    cbar_kwargs : dict, optional
        Keyword arguments for the colorbar, by default dict(orientation="vertical", label="Proportion [%]").
    cax_kwargs : dict, optional
        Keyword arguments for the inset axes where the colorbar will be placed, by default dict(width="5%", height="100%", loc="center left", bbox_to_anchor=(1.05, 0, 1, 1), borderpad=0).
    Returns
    -------
    axs : plt.Axes
        Matplotlib Axes object with the plot.
    cbar : plt.colorbar or None
        Matplotlib colorbar object if add_cbar is True, otherwise None.
    """

    df = trajectories.copy()
    # We assume hue_col is a str column, so first assign str values
    # and then convert to float
    # This avoids pd FutureWarning (see https://github.com/pandas-dev/pandas/issues/57734)
    df = df.replace({hue_col: {k: str(v) for k, v in zip(hue_order, hue_labels)}})
    df[hue_col] = df[hue_col].astype(float)

    # Plot the three variables
    sns.lineplot(
        data=df,
        x="level",
        y=quantity,
        hue=hue_col,
        hue_order=hue_labels,
        ax=axs,
        estimator=estimator,
        legend=False,
        palette=cmap,
        hue_norm=norm,
        zorder=10,
        **lineplot_kwargs,
    )

    if add_cbar:
        cax = inset_axes(axs, bbox_transform=axs.transAxes, **cax_kwargs)

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, **cbar_kwargs)
    else:
        cbar = None

    return axs, cbar


def plot_1d_histograms(
    dataframes: list[pl.DataFrame],
    quantities: list[str],
    labels: list[str],
    colors: list[str],
    axs: list[plt.Axes],
    hist_alpha=0.2,
    hist_lw=1.5,
    histogram_limits: dict = quantity_definitions.HISTOGRAM_LIMITS,
    histogram_nbins: dict = quantity_definitions.HISTOGRAM_NBINS,
    histogram_discrete: dict = quantity_definitions.HISTOGRAM_DISCRETE,
    histogram_ax_limits: dict = quantity_definitions.HISTOGRAM_AX_LIMITS,
    qty_titles: dict = quantity_definitions.TITLES,
    qty_formats: dict = quantity_definitions.QTY_FORMATS,
    rasterized=True,
):
    """Plot 1D histograms for each quantity in the dataframes.

    Parameters
    ----------
    dataframes : list[pl.DataFrame]
        List of polars DataFrames containing the data to plot. Each dataframe should have the quantities to plot as columns.
    quantities : list[str]
        List of quantities to plot. The quantities should be columns in the dataframes.
    labels : list[str]
        List of labels for each dataframe, used for the legend.
    colors : list[str]
        List of colors for each label, used for the plot.
    axs : list[plt.Axes]
        List of matplotlib Axes objects where the plots will be drawn. Needs to have at least as many axes as there are quantities.
    hist_alpha : float, optional
        Alpha value for the histogram fills, by default 0.2.
    hist_lw : float, optional
        Line width for the histogram edges, by default 1.5.
    histogram_limits : dict, optional
        Dictionary specifying the limits for each histogram. Keys are quantities and values are tuples (min, max).
    histogram_nbins : dict, optional
        Dictionary specifying the number of bins for each histogram. Keys are quantities and values are integers.
    histogram_discrete : dict, optional
        Dictionary specifying whether the histogram is discrete for each quantity. Keys are quantities and values are booleans.
    histogram_ax_limits : dict, optional
        Dictionary specifying the x-axis limits for each histogram. Keys are quantities and values are tuples (min, max).
    qty_titles : dict, optional
        Dictionary specifying the titles for each quantity. Keys are quantities and values are strings.
    qty_formats : dict, optional
        Dictionary specifying the format for each quantity's x-axis labels. Keys are quantities and values are format strings.

    Returns
    -------
    axs : list[plt.Axes]
        List of matplotlib Axes objects with the plots.

    """
    # Plot histograms for each quantity
    points = defaultdict(dict)
    for i, qty in enumerate(quantities):
        for df, label, color in zip(dataframes, labels, colors):
            df_ = df.select(qty).to_pandas().dropna()
            num_obs = df.count()[qty].item()
            # print(f"Plotting {qty} for {label} with {num_obs:,d} observations")
            histg = sns.histplot(
                data=df_,
                x=qty,
                ax=axs[i],
                stat="percent",
                color=color,
                # facecolor=colors["all"],
                edgecolor=color,
                element="step",
                linewidth=hist_lw,
                alpha=hist_alpha,
                bins=histogram_nbins[qty],
                binrange=histogram_limits[qty],
                discrete=histogram_discrete[qty],
                label=f"{label} (N = {num_obs:,d})",
                legend=True,
                rasterized=rasterized,
            )
            points_ = histg.get_children()[0].get_paths()[0].vertices
            num_points = points_.shape[0]
            xs = points_[num_points // 2 :, 0][::-1]
            ys = points_[num_points // 2 :, 1][::-1]
            points[qty][str(label)] = {"x": xs.tolist(), "y": ys.tolist()}

        axs[i].set_title(f"{qty_titles[qty]}")
        axs[i].set_xlabel(qty)
        axs[i].set_ylabel("Proportion [%]")
        axs[i].grid(axis="y", which="major")
        axs[i].grid(axis="y", which="minor")

        axs[i].set_xlim(histogram_ax_limits[qty])
        axs[i].set_xlabel(f"{qty_titles[qty]}")
        axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(qty_formats[qty]))

        axs[i].legend(
            loc="best",
            fontsize="small",
            framealpha=0.8,
            edgecolor="black",
            facecolor="white",
        )

    return axs, points


def plot_2d_histograms(
    dataframes: list[pl.DataFrame],
    labels: list[str],
    axs: list[plt.Axes],
    fig: plt.Figure,
    x_quantity: str,
    y_quantity: str,
    cmap=None,
    norm=None,
    histogram_limits: dict = quantity_definitions.HISTOGRAM_LIMITS,
    histogram_nbins: dict = quantity_definitions.HISTOGRAM_NBINS,
    histogram_ax_limits: dict = quantity_definitions.HISTOGRAM_AX_LIMITS,
    qty_titles: dict = quantity_definitions.TITLES,
    qty_formats: dict = quantity_definitions.QTY_FORMATS,
    cbar_ax=None,
    pthresh=None,
):
    """Plot 2D histograms for the specified quantity pairs in the dataframes.


    Parameters
    ----------
    dataframes : list[pl.DataFrame]
        List of polars DataFrames containing the data to plot. Each dataframe should have the quantities to plot as columns.
    labels : list[str]
        List of labels for each dataframe, used for the legend.
    axs : list[plt.Axes]
        List of matplotlib Axes objects where the plots will be drawn. Needs to have at least as many axes as there are dataframes.
    fig : plt.Figure
        Matplotlib Figure object to use for the colorbar.
    x_quantity : str
        The quantity to plot on the x-axis. This should be a column in the dataframes.
    y_quantity : str
        The quantity to plot on the y-axis. This should be a column in the dataframes.
    cmap : str or colormap, optional
        Colormap to use for the hue, by default None. If None, a default colormap will be used.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance to scale the colormap, by default None. If None, a default normalization will be used.
    histogram_limits : dict, optional
        Dictionary specifying the limits for each histogram. Keys are quantities and values are tuples (min, max).
    histogram_nbins : dict, optional
        Dictionary specifying the number of bins for each histogram. Keys are quantities and values are integers.
    histogram_ax_limits : dict, optional
        Dictionary specifying the x and y-axis limits for each histogram. Keys are quantities and values are tuples (min, max).
    qty_titles : dict, optional
        Dictionary specifying the titles for each quantity. Keys are quantities and values are strings.
    qty_formats : dict, optional
        Dictionary specifying the format for each quantity's x and y-axis labels. Keys are quantities and values are format strings.

    Returns
    -------
    axs : list[plt.Axes]
        List of matplotlib Axes objects with the plots.

    """

    # Plot 2D histograms for the quantity pairs
    for i, (df, label) in enumerate(zip(dataframes, labels)):
        sns.histplot(
            # data=ds_,
            x=df[x_quantity],
            y=df[y_quantity],
            ax=axs[i],
            binrange=[histogram_limits[x_quantity], histogram_limits[y_quantity]],
            bins=[histogram_nbins[x_quantity], histogram_nbins[y_quantity]],
            stat="percent",
            # palette=cmap,
            # hue_norm=norm,
            vmin=None,
            vmax=None,
            cmap=cmap,
            norm=norm,
            rasterized=True,
            pthresh=pthresh,
        )
        axs[i].set_title(label)

    # colorbar in last axis
    if cbar_ax is not None:
        cax = cbar_ax
    else:
        cax = inset_axes(
            axs[-1],
            width="5%",  # width of the colorbar
            height="100%",  # height of the colorbar
            loc="center left",
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=axs[-1].transAxes,
            borderpad=0,
        )

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
    cbar.set_label(
        "Proportion [%]",
    )

    for ax in axs.flatten():
        ax.set_xlabel(qty_titles[x_quantity])
        ax.set_ylabel(qty_titles[y_quantity])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(qty_formats[x_quantity]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(qty_formats[y_quantity]))
        # ax.yaxis.set_major_formatter(
        #     ticker.FuncFormatter())
        # )
        ax.grid(axis="both", which="major")
        ax.grid(axis="both", which="minor")
        ax.set_xlim(histogram_ax_limits[x_quantity])
        ax.set_ylim(histogram_ax_limits[y_quantity])

    return axs, cbar
