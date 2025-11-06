"""Create data that represents echo top height over melting level.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""

import argparse
import logging
import logging.config
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
import time
from collections import OrderedDict

import dask
import numpy as np
import pandas as pd
import rasterio as rio  # noqa
import rioxarray  # noqa
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from pyproj import CRS
from tqdm import tqdm

from utils.config_utils import load_config
from utils.data_utils import h5_to_dataset, metranet_to_dataset
from utils.date_utils import get_chunked_date_range

warnings.filterwarnings("ignore")

# Setup logging
with open("logconf.yaml", "rt") as f:
    log_config = yaml.safe_load(f.read())
    f.close()
logging.config.dictConfig(log_config)
logging.captureWarnings(True)
logger = logging.getLogger(Path(__file__).stem)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config", type=str, help="Configuration file path")
    argparser.add_argument("start_date", type=str, help="Start date of the interval, as YYYYmmddHHMM.")
    argparser.add_argument("end_date", type=str, help="End date of the interval, as YYYYmmddHHMM.")
    argparser.add_argument("etops", type=str, nargs="+", help="Echo top setup names in config file")
    argparser.add_argument("--nworkers", type=int, default=1, help="Number of workers")
    argparser.add_argument("--nchunks", type=int, default=1, help="Number of chunks per worker")
    argparser.add_argument("--debug", action="store_true", help="Debug mode (default: False)")
    argparser.add_argument("--timedelta", type=int, default=5, help="Timestep in minutes")
    args = argparser.parse_args()

    confpath = Path(args.config).resolve()

    startdate = datetime.strptime(args.start_date, "%Y%m%d%H%M")
    enddate = datetime.strptime(args.end_date, "%Y%m%d%H%M")

    conf = load_config(confpath)

    # Read projection sample file to use as template
    proj_sample_ds = xr.open_dataset(conf.projection.sample_file)
    proj_sample_ds = proj_sample_ds.rio.write_crs(proj_sample_ds.spatial_ref.crs_wkt)

    def worker(starttime, endtime, etop_setup):
        # Find echo top files
        days = pd.date_range(starttime, endtime, freq="1D", inclusive="both").to_pydatetime().tolist()
        datasets = []
        filenames = []
        for curdate in days:
            # Get echo top files
            #
            filename_pattern = conf.echo_top[etop_setup].input.filename
            rootpath = Path(curdate.strftime(conf.echo_top[etop_setup].input.path))
            filename_glob = re.sub(
                "(%[%YyjmdHMS])+",
                "*",
                filename_pattern,
            )
            filetimes = pd.DataFrame(
                {
                    "time": [
                        datetime.strptime(
                            p.name.split(".")[0],
                            filename_pattern.split(".")[0],
                        ).replace(second=0)
                        for p in rootpath.glob(filename_glob)
                    ],
                    "filename": list(rootpath.glob(filename_glob)),
                }
            ).sort_values("time")
            filetimes = filetimes.loc[(filetimes.time >= starttime) & (filetimes.time <= endtime)]
            etop_files = OrderedDict({r[0].to_pydatetime(): r[1] for _, r in filetimes.iterrows()})

            model_ds = None
            prev_model_file = None

            for timestamp, etop_fn in etop_files.items():

                if not conf.replace:
                    # Check if file already exists
                    out_fn = Path(timestamp.strftime(conf.echo_top[etop_setup]["output"].path)) / timestamp.strftime(
                        conf.echo_top[etop_setup]["output"].filename
                    )
                    if out_fn.exists():
                        # logger.info(f"File {out_fn} already exists, skipping")
                        continue
                    logger.info(f"Processing timestep {timestamp}: {etop_fn}")

                # Read file
                if etop_fn.suffix == ".h5":
                    etop_ds_ = h5_to_dataset(etop_fn, var_name="HGHT")
                elif re.match(r".\d+", etop_fn.suffix) is not None:
                    # Metranet with no specified suffix, only number
                    etop_ds_ = metranet_to_dataset(etop_fn, var_name="HGHT")
                else:
                    logger.error(f"File {etop_fn} not supported, skipping!")
                    continue

                # Reproject to radar data projection
                etop_ds = etop_ds_.rio.reproject_match(proj_sample_ds)

                # Read melting level
                model_fn = Path(timestamp.strftime(conf.melting_level.inputpath)) / timestamp.strftime(
                    conf.melting_level.inputfilename
                )
                if model_fn != prev_model_file:
                    if conf.melting_level.type == "iso0":
                        # Get path to model file
                        # Read model data
                        try:
                            model_ds_ = xr.open_dataset(model_fn)
                        except FileNotFoundError:
                            logger.error(f"File {model_fn} not found, skipping timestep")
                            continue
                        except OSError as e:
                            logger.error(f"Error reading file {model_fn}: {e}")
                            continue
                        model_ds_ = model_ds_.rio.write_crs(model_ds_.spatial_ref.crs_wkt)
                        # Drop lon, lat
                        if "latitude" in model_ds_.data_vars:
                            model_ds_ = model_ds_.drop_vars(["latitude", "longitude"])
                        # Reproject to radar data projection
                        model_ds = model_ds_.rio.reproject_match(proj_sample_ds)
                    elif conf.melting_level.type == "metranet":
                        try:
                            model_ds_ = metranet_to_dataset(model_fn, var_name="HGHT")
                        except FileNotFoundError:
                            logger.error(f"File {model_fn} not found, skipping timestep")
                            continue
                        except AttributeError as e:
                            logger.error(f"Error reading file {model_fn}: {e}")
                            continue
                        # Reproject to radar data projection
                        model_ds = model_ds_.rio.reproject_match(proj_sample_ds)

                    else:
                        raise NotImplementedError(
                            "Melting level type {} not implemented".format(conf.melting_level.type)
                        )
                    prev_model_file = model_fn
                # Calculate echo top over melting level
                et_over_ml = etop_ds.copy()

                # Time difference between model and echo top
                et_time = pd.Timestamp(etop_ds.time.item())
                model_time = pd.Timestamp(model_ds.time.item())
                timediff = (et_time - model_time).total_seconds() / 60
                # We allow up to 59 minutes difference
                if timediff > 59 or timediff < 0:
                    logger.warning(
                        f"Time difference between echo top and model data is {timediff:.0f} minutes, skipping timestep"
                    )
                    sys.exit(-1)
                    continue
                # import ipdb

                # ipdb.set_trace()

                et_over_ml["HGHT"].values = (etop_ds.HGHT.data - model_ds.HGHT.data).astype(np.float32)
                et_over_ml = et_over_ml.rename(dict(HGHT=f"echotop_{etop_ds.HGHT.attrs['threshold']:.0f}_over_ml"))
                # Set timestamp as coordinate
                et_over_ml = et_over_ml.assign_coords(
                    {
                        "time": (
                            "time",
                            [timestamp],
                            et_over_ml.time.attrs,
                        )
                    }
                )
                # Set negative values to zero but retain nans
                # et_over_ml = et_over_ml.where(np.isnan(et_over_ml) | (et_over_ml > 0), 0)
                # Write projection
                et_over_ml = et_over_ml.rio.write_crs(proj_sample_ds.rio.crs.to_proj4())
                # import ipdb

                # ipdb.set_trace()
                # Set attributes for compression
                comp = dict(zlib=True, complevel=9, dtype="float32")
                et_over_ml.encoding.update(comp)
                et_over_ml[list(et_over_ml.data_vars.keys())[0]].encoding.update(comp)

                # Export to netcdf
                out_fn = Path(timestamp.strftime(conf.echo_top[etop_setup]["output"].path)) / timestamp.strftime(
                    conf.echo_top[etop_setup]["output"].filename
                )
                out_fn.parent.mkdir(parents=True, exist_ok=True)
                # et_over_ml.to_netcdf(out_fn)

                datasets.append(et_over_ml)
                filenames.append(out_fn)

        if len(datasets) == 0:
            logger.warning(f"No echo top files created for {etop_setup} {starttime} - {endtime}")
            return
        try:
            xr.save_mfdataset(datasets, filenames, mode="w")
        except PermissionError as e:
            # Sometimes the writing fails, try again
            time.sleep(5)
            try:
                xr.save_mfdataset(datasets, filenames, mode="w")
            except PermissionError as e:
                logger.error(f"Error writing files for {etop_setup} {starttime} - {endtime}: {e}")
                return

    res = []

    td = pd.Timedelta(minutes=args.timedelta)

    date_ranges = get_chunked_date_range(startdate, enddate, num_chunks=args.nworkers * args.nchunks)

    for start, end in date_ranges:
        for etop_setup in args.etops:
            if args.nworkers == 1:
                worker(start, end, etop_setup)
            else:
                res.append(dask.delayed(worker)(start, end, etop_setup))

    if args.nworkers > 1:
        ProgressBar().register()
        dask.compute(
            *res,
            num_workers=args.nworkers,
            scheduler="processes",
            chunksize=1,
        )
