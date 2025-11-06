"""Script to check if all input data exists for times where the main variable exists."""

import argparse
import logging
import logging.config
import sys
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import xarray as xr

from utils.config_utils import load_config


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config", type=str, help="Configuration file directory")
    argparser.add_argument("start", type=str, help="Start date in YYYYMMDD format")
    argparser.add_argument("end", type=str, help="End date in YYYYMMDD format")
    argparser.add_argument("bad_date_file", type=str, help="File where missing times are written to.")
    argparser.add_argument("--timedelta", type=int, help="Time delta in minutes", default=5)

    args = argparser.parse_args()
    radar = None

    confpath = Path(args.config).resolve()

    conf = load_config(confpath)

    bad_times_file = Path(args.bad_date_file)
    bad_times_file.parents[0].mkdir(parents=True, exist_ok=True)
    # # Write header
    # with open(bad_times_file, "w") as f:
    #     f.write("missing_time,missing_var,notes\n")

    times = pd.date_range(
        start=pd.Timestamp(args.start).floor("D"),
        end=pd.Timestamp(args.end).ceil("D"),
        freq=f"{args.timedelta}min",
    ).to_pydatetime()

    times = [t for t in times if t.month in [5, 6, 7, 8, 9]]

    # import ipdb

    # ipdb.set_trace()

    variables = set(conf.keys()) - set(["common"])
    availability = pd.DataFrame(index=times, columns=list(variables))

    for variable in tqdm(variables):
        tqdm.write(f"Checking {variable}")
        dconf = conf[variable]
        for time in tqdm(times):
            # Check if the file exists
            path = Path(time.strftime(dconf.path))
            glob = time.strftime(dconf.filename)

            files = list(path.glob(glob))
            if len(files) == 0:
                availability.loc[time, variable] = False
                # # tqdm.write(f"File {path}/{glob} does not exist")
                # with open(bad_times_file, "a") as f:
                #     f.write(f"{time},{variable},\n")
                # continue
            # else:
            #     if "ml" in variable:
            #         # Try to open the file and check if it is empty
            #         try:
            #             ds = xr.open_dataset(files[0], cache=False)
            #         except Exception as e:
            #             tqdm.write(f"File {files[0]} cannot be opened, deleting it")
            #             # File is empty or cannot be opened
            #             # delete the file
            #             files[0].unlink()
            #             availability.loc[time, variable] = False

    availability.dropna(how="all", inplace=True)
    availability.to_csv(bad_times_file, header=True)
