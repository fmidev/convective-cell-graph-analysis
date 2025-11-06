import argparse
from pathlib import Path
import dask.bag as db
import dask
import os
import time
from datetime import datetime, timedelta
import pandas as pd

PYTHONPATH = Path(__file__).resolve().parents[1]

# SCRIPT = Path("identify_storm_cells.py")
# SCRIPT = Path("track_storm_cells.py")
# SCRIPT = Path("extract_cell_raster_stats.py")
RUNPATH = Path("stormcell_nowcasting/stormcell_nowcasting/").absolute()
SCRIPT = Path("extract_cell_raster_stats.py")
PYTHONPATH = Path("stormcell_nowcasting/").absolute()
CONFIG = "swiss-data-052025"
CMD_FMT = "cd {RUNPATH}; PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {starttime} {endtime} {CONFIG}; cd -"

# SCRIPT = Path("scripts/dataprocessing/query_storm_tracks.py")
# CONFIG = "config/plots/swiss-data/track_queries.yml"
# CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {CONFIG} {starttime} {endtime}"


# SCRIPT = Path("scripts/dataprocessing/filter_zdcol.py")
# CONFIG = Path("config/plots/swiss-data/zdrcol_filtering.yml")
# CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {starttime} {endtime} {CONFIG}"

# SCRIPT = Path("scripts/dataprocessing/calculate_echotop_over_melting_level.py")
# CONFIG = "config/plots/swiss-data/echotops.yml"
# CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {CONFIG} {starttime} {endtime} etop15 etop20 etop45 "

datelist = Path("~/all_dates_052021_092023.txt").expanduser()
DATES = pd.read_csv(datelist, header=None)[0].to_list()
# DATES = ["2021-05-30"]

DATES = [
    "2021-05-12",
    "2021-05-13",
    "2021-05-14",
    "2021-05-16",
    "2021-05-17",
    "2021-05-18",
    "2021-05-19",
    "2021-05-21",
    "2021-05-22",
    "2021-05-23",
    "2021-05-24",
    "2021-05-27",
    "2021-05-29",
    "2021-05-30",
    "2021-06-21",
    "2021-06-27",
    "2021-06-30",
    "2021-07-06",
    "2021-07-12",
    "2021-07-13",
    "2021-07-18",
    "2021-07-19",
    "2021-07-20",
    "2021-07-24",
    "2021-07-25",
    "2021-07-26",
    "2021-07-27",
    "2021-07-30",
    "2021-07-31",
    "2021-08-01",
    "2021-08-02",
    "2021-08-03",
    "2021-08-04",
    "2021-08-05",
    "2021-08-06",
    "2021-08-07",
    "2021-08-08",
    "2021-08-09",
    "2021-08-10",
    "2021-08-11",
    "2021-08-12",
    "2021-08-14",
    "2021-08-15",
    "2021-08-16",
    "2021-08-21",
    "2021-08-22",
    "2021-08-23",
    "2021-08-24",
    "2021-08-25",
    "2021-08-27",
    "2021-08-28",
    "2021-08-29",
    "2021-08-30",
    "2021-08-31",
    "2021-09-02",
    "2021-09-04",
    "2021-09-05",
    "2021-09-06",
    "2021-09-09",
    "2021-09-10",
    "2021-09-11",
    "2021-09-14",
    "2021-09-15",
    "2021-09-16",
    "2021-09-17",
    "2021-09-18",
    "2021-09-19",
    "2021-09-20",
    "2021-09-25",
    "2021-09-26",
    "2021-09-28",
    "2021-09-29",
    "2022-05-01",
    "2022-05-03",
    "2022-05-04",
    "2022-05-05",
    "2022-05-07",
    "2022-05-08",
    "2022-05-09",
    "2022-05-12",
    "2022-05-13",
    "2022-05-14",
    "2022-05-15",
    "2022-05-16",
    "2022-05-18",
    "2022-05-19",
    "2022-05-20",
    "2022-05-21",
    "2022-05-22",
    "2022-05-23",
    "2022-05-24",
    "2022-05-25",
    "2022-05-26",
    "2022-05-27",
    "2022-05-28",
    "2022-05-29",
    "2022-05-31",
    "2022-06-01",
    "2022-06-02",
    "2022-06-03",
    "2022-06-04",
    "2022-06-05",
    "2022-06-06",
    "2022-06-07",
    "2022-06-08",
    "2022-06-09",
    "2022-06-12",
    "2022-06-13",
    "2022-06-14",
    "2022-06-15",
    "2022-06-16",
    "2022-06-18",
    "2022-06-19",
    "2022-06-20",
    "2022-06-21",
    "2022-06-22",
    "2022-06-23",
    "2022-06-24",
    "2022-06-25",
    "2022-06-26",
    "2022-06-27",
    "2022-06-28",
    "2022-06-29",
    "2022-06-30",
    "2022-07-01",
    "2022-07-03",
    "2022-07-04",
    "2022-07-05",
    "2022-07-06",
    "2022-07-07",
    "2022-07-11",
    "2022-07-12",
    "2022-07-14",
    "2022-07-15",
    "2022-07-16",
    "2022-07-17",
    "2022-07-22",
    "2022-07-23",
    "2022-07-24",
    "2022-07-25",
    "2022-07-26",
    "2022-07-27",
    "2022-07-28",
    "2022-07-29",
    "2022-07-30",
    "2022-08-01",
    "2022-08-02",
    "2022-08-04",
    "2022-08-05",
    "2022-08-06",
    "2022-08-08",
    "2022-08-09",
    "2022-08-10",
    "2022-08-11",
    "2022-08-12",
    "2022-08-13",
    "2022-08-14",
    "2022-08-15",
    "2022-08-16",
    "2022-08-17",
    "2022-08-18",
    "2022-08-19",
    "2022-08-20",
    "2022-08-22",
    "2022-08-25",
    "2022-08-26",
    "2022-08-27",
    "2022-08-28",
    "2022-08-29",
    "2022-08-30",
    "2022-08-31",
    "2022-09-01",
    "2022-09-02",
    "2022-09-03",
    "2022-09-04",
    "2022-09-05",
    "2022-09-06",
    "2022-09-07",
    "2022-09-08",
    "2022-09-09",
    "2022-09-10",
    "2022-09-11",
    "2022-09-13",
    "2022-09-14",
    "2022-09-15",
    "2022-09-16",
    "2022-09-17",
    "2022-09-23",
    "2022-09-24",
    "2023-09-13",
    "2023-09-22",
]

PLUS_MINS = 60 * 24
MINUS_MINS = 0
# DATES = [
#     # "202308280000",
#     # "202308270000",
#     # "202105100000",
#     "202107130000",
#     # "202107080000",
#     # "202105110000",
#     # "202208180000",
#     # "202309220000",
#     # "202209280000",
#     # "202307240000",
# ]


# # Plot case gifs
# SCRIPT = Path("scripts/viz/plot_data_gifs.py")
# CONFIG = "config/plots/swiss-data/plot_data.yml"
# CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {CONFIG} {starttime}"

# # SCRIPT = Path("scripts/viz/plot_data_comparison.py")
# # CONFIG = "config/plots/swiss-data/plot_data.yml"
# # CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {CONFIG} {starttime} {endtime} --outdir /data/jenna/rain-cell-stats-analysis/figures/data_histograms_v_20250321 --vars zdr_column_grid zdr_column_zh VIL ET45ML ET20ML ET15ML FZT MaxEcho RHOHV "

# # SCRIPT = Path("scripts/dataprocessing/filter_zdcol.py")
# # CONFIG = "config/plots/swiss-data/zdrcol_filtering.yml"
# # CMD_FMT = "PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {starttime} {endtime} {CONFIG} "


# PLUS_MINS = 60 * 3
# MINUS_MINS = 0
# DATES = [
# "202308280900",
# "202308272100",
# "202105100800",
# "202107131200",
# "202107081500",
# "202105110300",
# "202208180800",
# "202309221500",
# "202209281200",
# "202307240300",
# ]

if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--nworkers", type=int, default=1, help="Number of workers.")

    args = parser.parse_args()

    def worker(command):
        # import ipdb

        # ipdb.set_trace()
        print(f"Running command: {command}")
        # return
        os.system(command)
        # time.sleep(3)
        return

    commands = []

    all_timesteps = []
    for timestamp in DATES:
        # date = datetime.strptime(timestamp, "%Y%m%d%H%M")
        date = datetime.strptime(timestamp, "%Y-%m-%d")
        start = date - timedelta(minutes=MINUS_MINS)
        starttime = start.strftime("%Y%m%d%H%M")
        end = date + timedelta(minutes=PLUS_MINS)
        endtime = end.strftime("%Y%m%d%H%M")

        # command = f"PYTHONPATH=$PYTHONPATH:{PYTHONPATH} python {SCRIPT} {starttime} {endtime} {CONFIG}"
        command = CMD_FMT.format(
            PYTHONPATH=PYTHONPATH, SCRIPT=SCRIPT, starttime=starttime, endtime=endtime, CONFIG=CONFIG, RUNPATH=RUNPATH
        )
        commands.append(command)

    from dask.distributed import Client

    scheduler = "processes" if args.nworkers > 1 else "single-threaded"
    client = Client(n_workers=args.nworkers, threads_per_worker=1)

    bag = db.from_sequence(commands)
    # from dask.diagnostics import ProgressBar

    # ProgressBar().register()
    print(f"Using scheduler: {scheduler} with {args.nworkers} workers")
    # with dask.config.set(scheduler=scheduler, num_workers=args.nworkers):
    res = bag.map(worker).compute()
