"""Track storm cells in the given time range and insert their temporal
connectivity information to the database. Optical flow is applied to enhance
the robustness of the tracking.

Configuration files (in the config directory)
---------------------------------------------
- database.yaml
- raster_datasources.yaml
- track_storm_cells.yaml

Input
-----
- storm cells stored in the database (the stormcells table)
- rasters from the data source selected in track_storm_cells.yaml and defined
  in raster_datasources.yaml (for computing the optical flow)

Output
------
- temporal connectivity of storm cells inserted to the next_cells table
- merged, splitted, vel_x and vel_y columns in the stormcell_attributes table
  the velocities are estimated from the optical flow
  the unit of vel_x and vel_y is geographical distance unit / hour
- printing into terminal or to the log file specified in track_storm_cells.yaml
  if num_workers=1

Existing entries for the time stamps in the given range are overwritten.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import yaml
from affine import Affine
import numpy as np
from rasterstats import zonal_stats
from shapely import affinity
from shapely.errors import GEOSException
from sqlalchemy.dialects.postgresql import insert as sa_insert

from stormcell_nowcasting.common import database as database_methods, util
from stormcell_nowcasting.datasources import raster_datasources
from stormcell_nowcasting.common import oflow


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Track storm cells and store temporal connectivity information to database."
    )

    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    parser.add_argument("endtime", help="end time (YYYYmmddHHMM)")
    parser.add_argument("config", help="configuration profile to use")

    args = parser.parse_args()

    starttime = datetime.strptime(args.starttime, "%Y%m%d%H%M")
    endtime = datetime.strptime(args.endtime, "%Y%m%d%H%M")

    # read configuration files
    with open(os.path.join("config", args.config, "track_storm_cells.yaml"), "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join("config", args.config, "database.yaml"), "r") as f:
        config_database = yaml.safe_load(f)

    with open(os.path.join("config", args.config, "raster_datasources.yaml"), "r") as f:
        config_datasources = yaml.safe_load(f)

    timestep = config_datasources["common"]["timestep"]

    logger = util.get_logger(config["logging"]["level"], output_file=config["logging"]["output_file"])

    database = database_methods.get_backend(config_database["database_type"])

    def worker(starttime, endtime, cell_identification_method, cell_query_kwargs):
        # connect to the database
        engine = database.create_engine(**config_database)
        db_conn = engine.connect()

        # delete existing database entries in the time range to avoid creating
        # duplicates
        if cell_query_kwargs is not None:
            tracking_method_name = cell_identification_method + f":{cell_query_kwargs['exp_name']}"
            cq_kwargs = cell_query_kwargs["kwargs"]
        else:
            tracking_method_name = cell_identification_method
            cq_kwargs = {}
        if config["database"]["delete_existing"]:
            database.delete_rows(
                database.NextCellTable,
                starttime,
                endtime,
                db_conn,
                tracking_method_name,
            )

        curtime = starttime

        # query storm cells in the given time range and create a dictionary
        stormcell_list = database.query_storm_cells(
            starttime - timedelta(minutes=timestep),
            endtime + timedelta(minutes=timestep),
            db_conn,
            cell_identification_method,
            **cq_kwargs,
        )
        stormcells = defaultdict(list)
        for sc in stormcell_list:
            sc["bbox"] = sc.polygon.bounds
            sc["merged"] = False
            sc["splitted"] = False
            stormcells[sc.obstime].append(sc)

        prev_cells = defaultdict(lambda: defaultdict(list))
        next_cells = defaultdict(lambda: defaultdict(list))
        n_db_entries = 0
        commit_start_time = starttime
        while curtime <= endtime + timedelta(minutes=timestep):
            try:
                prevtime = curtime - timedelta(minutes=timestep)

                if len(stormcells[prevtime]) > 0 and len(stormcells[curtime]) > 0:
                    stormcells_found = True

                    if config["parallelization"]["num_workers"] == 1:
                        comp_starttime = time.time()

                    starttime_inputs = time.time()

                    input_fields = []
                    for i in range(2):
                        input_time = prevtime if i == 0 else curtime

                        field, _, affine = raster_datasources.get_data(
                            input_time,
                            **config_datasources[config["oflow"]["quantity"]],
                        )
                        input_fields.append(field)

                    if config["parallelization"]["num_workers"] == 1:
                        logger.info(
                            f"read input radar composites at {curtime} in {time.time() - starttime_inputs:.02f} seconds"
                        )

                    # compute advection field by using optical flow
                    starttime_advfield = time.time()

                    if0 = input_fields[0].copy()
                    if0[~np.isfinite(if0)] = 0
                    if1 = input_fields[1].copy()
                    if1[~np.isfinite(if1)] = 0

                    advfield = oflow.compute_advection_field(
                        if0,
                        if1,
                        pyr_scale=config["oflow_params"]["pyr_scale"],
                        levels=config["oflow_params"]["levels"],
                        winsize=config["oflow_params"]["winsize"],
                        iterations=config["oflow_params"]["iterations"],
                        poly_n=config["oflow_params"]["poly_n"],
                        poly_sigma=config["oflow_params"]["poly_sigma"],
                        filter_stddev=config["oflow_params"]["filter_stddev"],
                        minval=config["oflow_params"]["minval"],
                    )
                    advfield[:, :, 0] *= affine[0, 1] * 60.0 / timestep
                    advfield[:, :, 1] *= affine[1, 2] * 60.0 / timestep

                    if config["parallelization"]["num_workers"] == 1:
                        logger.info(
                            f"computed advection field at {curtime} in "
                            f"{time.time() - starttime_advfield:02f} seconds"
                        )

                    def compute_adv_vel_means(stormcells, i):
                        return zonal_stats(
                            [stormcells[j].polygon for j in range(len(stormcells))],
                            advfield[:, :, i],
                            affine=Affine(*np.roll(affine, -1, axis=1).flatten()),
                            stats=["mean"],
                            nodata=np.nan,
                        )

                    # compute mean advection velocities inside the storm cells
                    starttime_adv_vel = time.time()

                    for i in range(2):
                        zs = compute_adv_vel_means(stormcells[curtime], i)
                        key = "vel_x" if i == 0 else "vel_y"
                        for j in range(len(stormcells[curtime])):
                            stormcells[curtime][j][key] = zs[j]["mean"]

                    if config["parallelization"]["num_workers"] == 1:
                        logger.info(
                            f"computed storm advection velocities at"
                            f" {curtime} in {time.time() - starttime_adv_vel:02f} seconds"
                        )

                    stormcells_cur = [stormcells[prevtime]] + [stormcells[curtime]]

                    storm_vels = []
                    for i in range(2):
                        zs = compute_adv_vel_means(stormcells_cur[0], i)
                        storm_vels.append([zs[j]["mean"] for j in range(len(zs))])

                    sc2_polys = []
                    for i in range(len(stormcells_cur[0])):
                        sc2_polys.append(
                            affinity.translate(
                                stormcells_cur[0][i].polygon,
                                xoff=storm_vels[0][i] / 60.0 * timestep,
                                yoff=storm_vels[1][i] / 60.0 * timestep,
                            )
                        )

                    for i in range(len(stormcells_cur[1])):
                        # sc1 is the current timestep cell
                        sc1 = stormcells_cur[1][i]

                        for j in range(len(stormcells_cur[0])):
                            # sc2 is the unshifted previous timestep cell
                            sc2 = stormcells_cur[0][j]
                            sc1_poly = sc1.polygon
                            # shift the previous storm cell forward in time by
                            # using the mean advection velocities inside the cells
                            # this improves the reliability of the matching between
                            # time steps
                            if _bbox_intersection(sc1["bbox"], sc2_polys[j].bounds):
                                success = True
                                try:
                                    isct_poly = sc1.polygon.intersection(sc2_polys[j])
                                except GEOSException:
                                    try:
                                        poly1 = sc1.polygon.buffer(0)
                                        poly2 = sc2_polys[j].buffer(0)
                                        isct_poly = poly1.intersection(poly2)
                                    except GEOSException:
                                        logger.error(
                                            f"failed to calculate intersection "
                                            f"between cells {sc1.identifier} and "
                                            f"{sc2.identifier} at {curtime}"
                                        )
                                        success = False

                                if success:
                                    overlap_area = isct_poly.area
                                    overlap_pct = overlap_area / sc1_poly.area
                                    # overlap_pct = overlap_area / min(sc1_poly.area, sc2_polys[j].area)
                                    if overlap_area > config["overlap_thr"] * min(sc1_poly.area, sc2_polys[j].area):
                                        prev_cells[curtime][sc1.identifier].append((sc2.identifier, overlap_pct))
                                        if len(prev_cells[curtime][sc1.identifier]) > 1:
                                            sc1["merged"] = True
                                        if prevtime >= starttime:
                                            next_cells[prevtime][sc2.identifier].append((sc1.identifier, overlap_pct))
                                            if len(next_cells[prevtime][sc2.identifier]) > 1:
                                                sc2["splitted"] = True
                                        n_db_entries += 1
                else:
                    stormcells_found = False

                if config["parallelization"]["num_workers"] == 1:
                    if stormcells_found:
                        logger.info(
                            f"total time for tracking storm cells at {curtime}: "
                            f"{time.time() - comp_starttime:.2f} seconds"
                        )
                    else:
                        logger.info(f"no storm cells found at {curtime}")

                # write connectivity information into database if the buffer is full
                # or if we are at the endpoint of the time interval
                if n_db_entries >= config["database"]["commit_size"] or curtime == endtime + timedelta(
                    minutes=timestep
                ):
                    if config["parallelization"]["num_workers"] == 1:
                        comp_starttime = time.time()

                    _write_connectivity_to_database(
                        stormcells,
                        next_cells,
                        commit_start_time,
                        min(curtime, endtime),
                        db_conn,
                        database,
                        tracking_method_name,
                    )

                    if config["parallelization"]["num_workers"] == 1 and n_db_entries > 0:
                        logger.info(
                            f"inserted {n_db_entries} entries to database in "
                            f"{time.time() - comp_starttime:.2f} seconds"
                        )

                    prev_cells = defaultdict(lambda: defaultdict(list))
                    next_cells = defaultdict(lambda: defaultdict(list))

                    n_db_entries = 0
                    commit_start_time = curtime + timedelta(minutes=timestep)
            except FileNotFoundError as e:
                if config["parallelization"]["num_workers"] == 1:
                    logger.error(e)
            except ModuleNotFoundError as e:
                if config["parallelization"]["num_workers"] == 1:
                    logger.error(e)
            except Exception as e:
                if config["parallelization"]["num_workers"] == 1:
                    logger.error("unspecified error: " + str(curtime))
                    logger.debug(e)

            curtime = curtime + timedelta(minutes=timestep)

        # close the database connection
        db_conn.close()
        engine.dispose()

    for method in config["cell_identification_method"]:
        for cell_query_kwargs in config["cell_query_kwargs"]:
            util.process_chunked_time_range(
                starttime,
                endtime,
                timestep,
                worker,
                method,
                cell_query_kwargs,
                num_workers=config["parallelization"]["num_workers"],
                max_chunk_len=config["parallelization"]["max_chunk_len"],
            )


def _bbox_intersection(bb1, bb2):
    if bb1[0] > bb2[2] or bb1[2] < bb2[0]:
        return False
    if bb1[1] > bb2[3] or bb1[3] < bb2[1]:
        return False

    return True


def _write_connectivity_to_database(
    cell_dict,
    next_cell_dict,
    starttime,
    endtime,
    db_conn,
    database_methods,
    cell_identification_method,
):
    values = []

    for analysistime in next_cell_dict.keys():
        for id1 in next_cell_dict[analysistime].keys():
            for id2, overlap_area in next_cell_dict[analysistime][id1]:
                values.append(
                    {
                        "timestamp": analysistime,
                        "identifier": id1,
                        "method": cell_identification_method,
                        "next_identifier": id2,
                        "overlap_area": overlap_area,
                    }
                )

    if len(values) > 0:
        insert = database_methods.table_next_cells.insert().values(values)
        db_conn.execute(insert)
        db_conn.commit()

    values = []

    for analysistime in cell_dict.keys():
        if analysistime >= starttime and analysistime <= endtime:
            for cell in cell_dict[analysistime]:
                val_dict = {
                    "timestamp": analysistime,
                    "identifier": cell.identifier,
                    "method": cell_identification_method,
                    "merged": cell["merged"],
                    "splitted": cell["splitted"],
                }
                if "vel_x" in cell.attributes:
                    val_dict["vel_x"] = cell["vel_x"]
                else:
                    val_dict["vel_x"] = None
                if "vel_y" in cell.attributes:
                    val_dict["vel_y"] = cell["vel_y"]
                else:
                    val_dict["vel_y"] = None
                values.append(val_dict)

    if len(values) > 0:
        insert = sa_insert(database_methods.table_cell_attributes).values(values)
        index_elements = ["timestamp", "identifier", "method"]
        set_ = {
            c.name: c
            for c in insert.excluded
            if not c.primary_key and c.name in ["merged", "splitted", "vel_x", "vel_y"]
        }
        insert = insert.on_conflict_do_update(index_elements=index_elements, set_=set_)
        db_conn.execute(insert)
        db_conn.commit()


if __name__ == "__main__":
    main()
