"""Calculate zonal statistics of the identified storm cells in the given
time range. Insert the statistics to the database. The statistics are computed
by applying rasterstats.zonal_stats using the given raster data sources.

Configuration files (in the config directory)
---------------------------------------------
- database.yaml
- extract_cell_raster_stats.yaml
- raster_datasources.yaml

Input
-----
- storm cells in the database (the stormcells table) created by running
  identify_storm_cells.py
- the raster data sources listed in extract_cell_raster_stats.yaml and defined
  in raster_datasources.yaml

Output
------
- zonal statistics of storm cells inserted to the database
  (the stormcell_rasterstats table)
- printing into terminal or to the log file specified in
  extract_cell_raster_stats.yaml if num_workers=1
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import os
import yaml
import time
from affine import Affine
from geoalchemy2.shape import to_shape
import numpy as np
from rasterstats import zonal_stats
from sqlalchemy.dialects.postgresql import insert as sa_insert
from skimage.measure import label
from scipy import ndimage
import pandas as pd
from itertools import chain
from collections import Counter

from stormcell_nowcasting.common import database as database_methods, util
from stormcell_nowcasting.datasources import raster_datasources


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute zonal statistics of storm cells for the user-supplied raster data sources."
    )

    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    parser.add_argument("endtime", help="end time (YYYYmmddHHMM)")
    parser.add_argument("config", help="configuration profile to use")

    args = parser.parse_args()

    starttime = datetime.strptime(args.starttime, "%Y%m%d%H%M")
    endtime = datetime.strptime(args.endtime, "%Y%m%d%H%M")

    # read configuration files
    with open(os.path.join("config", args.config, "extract_cell_raster_stats.yaml"), "r") as f:
        config = yaml.safe_load(f)

    output_quantities = config["output"]["quantities"]  # .split(",")
    output_stats = config["output"]["stats"].split(",")

    with open(os.path.join("config", args.config, "database.yaml"), "r") as f:
        config_database = yaml.safe_load(f)

    with open(os.path.join("config", args.config, "raster_datasources.yaml"), "r") as f:
        config_datasources = yaml.safe_load(f)

    timestep = config_datasources["common"]["timestep"]

    # create logger
    logger = util.get_logger(config["logging"]["level"], output_file=config["logging"]["output_file"])

    database = database_methods.get_backend(config_database["database_type"])

    def worker(starttime, endtime, cell_method):
        # connect to the database
        engine = database.create_engine(**config_database)
        db_conn = engine.connect()

        # query storm cells from the database in the given time range
        query = database.table_stormcells.select().where(
            database.table_stormcells.c.timestamp >= (starttime - timedelta(minutes=timestep)),
            database.table_stormcells.c.timestamp <= endtime,
            database.table_stormcells.c.method == cell_method,
        )
        result = db_conn.execute(query)

        stormcells = defaultdict(list)
        for row in result:
            analysistime = row[0]
            identifier = row[1]
            coords = row[3]
            poly = to_shape(coords)
            stormcells[analysistime].append(
                {
                    "id": identifier,
                    "polygon": poly,
                }
            )

        curtime = starttime

        db_values = []

        while curtime <= endtime:
            if config["parallelization"]["num_workers"] == 1:
                logger.info(
                    "started raster statistics computation at {} for {} cells".format(curtime, len(stormcells[curtime]))
                )

            # go through the requested quantities, retrieve input rasters and
            # compute the requested zonal statistics for each storm polygon
            db_entries_inserted = False

            for quantity in output_quantities:
                try:
                    if config["parallelization"]["num_workers"] == 1:
                        starttime = time.time()

                    # read the input raster, projection and transformation
                    input_field, projection, affine = raster_datasources.get_data(
                        curtime, **config_datasources[quantity]
                    )

                    if (
                        "quantities_from_overlapping_areas" in config["output"]
                        and quantity in config["output"]["quantities_from_overlapping_areas"].keys()
                    ):
                        conf_ = config["output"]["quantities_from_overlapping_areas"][quantity]
                        # if the quantity is from overlapping areas, we need to
                        # set the nodata value to 0
                        threshold = conf_["threshold"]
                        image = np.where(input_field > threshold, 1, 0)
                        # dilate image
                        kernel_size = conf_["dilation_kernel_size"]
                        kernel = np.ones(
                            (kernel_size, kernel_size),
                            dtype=np.uint8,
                        )
                        image = ndimage.binary_dilation(image, structure=kernel)
                        labeled_image = label(label_image=image, connectivity=2)

                        # this will have in ith element a dictionary with
                        # the label as key and the number of pixels in the
                        # storm cell as value
                        labels_inside_cells = zonal_stats(
                            [sc["polygon"] for sc in stormcells[curtime]],
                            labeled_image,
                            affine=Affine(*np.roll(affine, -1, axis=1).flatten()),
                            stats=[],
                            nodata=0,
                            all_touched=conf_.get("zonal_stats_all_touched", True),
                            categorical=True,
                        )

                        if conf_.get("assign_to_unique_cell", False):
                            # If the quantity is overlapping with multiple cells,
                            # we assign the value to the cell with the most pixels
                            cell_counts = Counter(chain(*[list(d.keys()) for d in labels_inside_cells]))
                            # Subract one for each label, so the resulting
                            # cell_counts will only have have with count larger than 1
                            one_counts = Counter(range(labeled_image.max() + 1))
                            cell_counts -= one_counts
                            for qlabel, _ in cell_counts.items():
                                # the label is in multiple cells, so we
                                # assign it to the cell with the most pixels
                                max_cell = max(
                                    labels_inside_cells,
                                    key=lambda d: d.get(qlabel, 0),
                                )
                                max_cell_i = labels_inside_cells.index(max_cell)
                                for i, d in enumerate(labels_inside_cells):
                                    if qlabel in d and i != max_cell_i:
                                        d.pop(qlabel, None)

                        stats = []
                        for i in range(len(stormcells[curtime])):
                            if not labels_inside_cells[i]:
                                stats.append({k: None for k in output_stats})
                            else:
                                values = input_field[
                                    (input_field > threshold)
                                    & np.isin(labeled_image, list(labels_inside_cells[i].keys()))
                                ]
                                stats.append(_get_stats(values, output_stats))
                    else:

                        # compute the zonal statistics
                        stats = zonal_stats(
                            [sc["polygon"] for sc in stormcells[curtime]],
                            input_field,
                            affine=Affine(*np.roll(affine, -1, axis=1).flatten()),
                            stats=output_stats,
                            nodata=np.nan,
                            all_touched=True,
                        )

                    # construct the data structure for inserting the values to
                    # database
                    for i in range(len(stats)):
                        for k in stats[i].keys():
                            if stats[i][k] is not None:
                                vd = {
                                    "timestamp": curtime,
                                    "identifier": stormcells[curtime][i]["id"],
                                    "method": cell_method,
                                    "quantity": quantity,
                                    "statistic": k,
                                    "value": stats[i][k],
                                }
                                db_values.append(vd)

                    if config["parallelization"]["num_workers"] == 1:
                        logger.info(
                            "processed raster for quantity '{}' in {:.2f} seconds.".format(
                                quantity, time.time() - starttime
                            )
                        )
                except FileNotFoundError as e:
                    if config["parallelization"]["num_workers"] == 1:
                        logger.error(e)
                except ModuleNotFoundError as e:
                    if config["parallelization"]["num_workers"] == 1:
                        logger.error(e)
                # except Exception as e:
                #     if config["parallelization"]["num_workers"] == 1:
                #         logger.error("unspecified error")
                #         logger.debug(e)

                # write the entries to database if the buffer is full or if we are
                # at the endpoint of the time interval
                if len(db_values) >= config["database"]["commit_size"] or curtime == endtime:
                    if len(db_values) > 0:
                        insert = sa_insert(database.table_cell_raster_stats).values(db_values)
                        index_elements = [
                            "timestamp",
                            "identifier",
                            "method",
                            "quantity",
                            "statistic",
                        ]
                        set_ = {c.name: c for c in insert.excluded if not c.primary_key and c.name in ["value"]}
                        insert = insert.on_conflict_do_update(index_elements=index_elements, set_=set_)
                        db_conn.execute(insert)
                        db_entries_inserted = True
                        db_values = []

            if db_entries_inserted:
                db_conn.commit()
                if config["parallelization"]["num_workers"] == 1:
                    logger.info("committed entries to the database")

            curtime = curtime + timedelta(minutes=timestep)

        # close the database connection
        db_conn.close()
        engine.dispose()

    for cell_method in config["cell_identification_methods"]:
        # process the time range in chunks
        util.process_chunked_time_range(
            starttime,
            endtime,
            timestep,
            worker,
            num_workers=config["parallelization"]["num_workers"],
            max_chunk_len=config["parallelization"]["max_chunk_len"],
            cell_method=cell_method,
        )


def _get_stats(arr, stats):
    feature_stats = {}
    if "min" in stats:
        feature_stats["min"] = float(arr.min())
    if "max" in stats:
        feature_stats["max"] = float(arr.max())
    if "mean" in stats:
        feature_stats["mean"] = float(arr.mean())
    if "count" in stats:
        feature_stats["count"] = int(arr.size)
    # optional
    if "sum" in stats:
        feature_stats["sum"] = float(arr.sum())
    if "std" in stats:
        feature_stats["std"] = float(arr.std())
    if "median" in stats:
        feature_stats["median"] = float(np.median(arr))
    if "unique" in stats:
        feature_stats["unique"] = len(np.unique(arr))
    if "range" in stats:
        try:
            rmin = feature_stats["min"]
        except KeyError:
            rmin = float(arr.min())
        try:
            rmax = feature_stats["max"]
        except KeyError:
            rmax = float(arr.max())
        feature_stats["range"] = rmax - rmin

    for pctile in [s for s in stats if s.startswith("percentile_")]:
        q = _get_percentile(pctile)
        pctarr = arr
        feature_stats[pctile] = float(np.percentile(pctarr, q))
    return feature_stats


def _get_percentile(stat):
    if not stat.startswith("percentile_"):
        raise ValueError("must start with 'percentile_'")
    qstr = stat.replace("percentile_", "")
    q = float(qstr)
    if q > 100.0:
        raise ValueError("percentiles must be <= 100")
    if q < 0.0:
        raise ValueError("percentiles must be >= 0")
    return q


if __name__ == "__main__":

    main()
