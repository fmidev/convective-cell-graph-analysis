"""Filter Zdr column data."""

import argparse
import os
from copy import copy
from datetime import datetime
from pathlib import Path
import gzip

import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

from utils.config_utils import load_config
from utils.data_utils import load_data
from utils.date_utils import get_chunked_date_range


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("startdate", type=str, help="Start date in YYYYMMDD format")
    argparser.add_argument("enddate", type=str, help="End date in YYYYMMDD format")
    argparser.add_argument("config", type=str, help="Path to configuration file")
    argparser.add_argument("--nworkers", type=int, default=1, help="Number of workers")
    argparser.add_argument("--nchunks", type=int, default=1, help="Number of chunks")
    argparser.add_argument("--timedelta", type=int, default=5, help="Time delta in minutes")
    args = argparser.parse_args()

    startdate = datetime.strptime(args.startdate, "%Y%m%d%H%M")
    enddate = datetime.strptime(args.enddate, "%Y%m%d%H%M")
    config = load_config(args.config)

    date_ranges = get_chunked_date_range(startdate, enddate, num_chunks=args.nworkers * args.nchunks)

    dataconf = copy(config.filtering_data)
    dataconf["input"] = config.input

    def worker(start, end):

        timesteps = pd.date_range(start, end, freq=f"{args.timedelta}min")
        # import ipdb

        # ipdb.set_trace()
        for timestep in timesteps:

            outpath = Path(timestep.strftime(config.output.path))
            outpath.mkdir(parents=True, exist_ok=True)
            filename = outpath / timestep.strftime(config.output.file)

            if filename.exists() and not config.output.overwrite:
                continue

            # Load data for timestep
            try:
                ds = load_data(
                    dataconf,
                    timestep,
                    [timestep],
                    1,
                    None,
                )
            except ValueError:
                print(f"Data not found for {timestep}")
                continue

            for i in range(1, len(config.filtering_conditions.keys()) + 1):
                condition = config.filtering_conditions[f"cond_{i}"]
                if condition["type"] == "accept_range":
                    # Accept only data within range
                    mask = np.ones(ds[config.input.variable].shape, dtype=bool)
                    if condition["min"] is not None:
                        mask = mask & (ds[condition.variable].data >= condition["min"])
                    if condition["max"] is not None:
                        mask = mask & (ds[condition.variable].data <= condition["max"])
                    ds[config.input.variable] = ds[config.input.variable].where(mask)
                elif condition["type"] == "accept_smaller_than":
                    # Accept only data smaller than the variable value
                    mask = ds[config.input.variable].data <= ds[condition.variable].data
                    ds[config.input.variable] = ds[config.input.variable].where(mask)
                elif condition["type"] == "accept_greater_than":
                    # Accept only data greater than the variable value
                    mask = ds[config.input.variable].data >= ds[condition.variable].data
                    ds[config.input.variable] = ds[config.input.variable].where(mask)
                elif condition["type"] == "contiguity":

                    image = np.where(ds[config.input.variable] > 0, 1, 0)
                    labeled_image = label(label_image=image, connectivity=2)
                    valid_pixels = np.zeros_like(image)

                    for region in regionprops(labeled_image):
                        if region.area >= condition["pixel_count"]:
                            valid_pixels = np.where(labeled_image == region.label, 1, valid_pixels)

                    ds[config.input.variable] = ds[config.input.variable].where(valid_pixels)

                else:
                    raise ValueError(f"Unknown filtering condition type: {condition['type']}")

            arr = ds[config.input.variable].values.squeeze()

            # Save data
            with gzip.GzipFile(filename, "w") as f:
                np.save(file=f, arr=np.flipud(arr))

    res = []
    for start, end in date_ranges:
        if args.nworkers == 1:
            worker(start, end)
        else:
            res.append(dask.delayed(worker)(start, end))

    if args.nworkers > 1:
        ProgressBar().register()
        dask.compute(
            *res,
            num_workers=args.nworkers,
            scheduler="processes",
            chunksize=1,
        )
