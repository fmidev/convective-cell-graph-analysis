"""Identify storm cells from radar images in the given time range and insert
them to the database.

Configuration files (in the config directory)
---------------------------------------------
- database.yaml
- identify_storm_cells.yaml
- raster_datasources.yaml

Input
-----
- rasters from the data sources listed in identify_storm_cells.yaml and defined
  in raster_datasources.yaml

Output
------
- identified storm cells are inserted to the stormcells table
- centroid location and area inserted to the stormcell_attributes table
- If delete_existing=True in identify_storm_cells.yaml, all existing entries
  in the stormcells table in the specified time range are deleted. This is also
  done for all tables that depend on stormcells: next_cells,
  stormcell_attributes, stormcell_rasterstats, stormcell_approx_ellipse.
- printing into terminal or to the log file specified in identify_storm_cells.yaml
  if num_workers=1

The coordinates of the polygons and their centroids are in the projection read
from the input rasters.

Existing entries between the given start and end time are overwritten.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import product
import os
import time
import yaml
import geoalchemy2
import numpy as np
from pyproj.transformer import Transformer
import shapely
from shapely.ops import transform

from stormcell_nowcasting.cell_identification.clustering import distance_clustering
from stormcell_nowcasting.cell_identification import contour
from stormcell_nowcasting.datasources import raster_datasources
from stormcell_nowcasting.common import database as database_methods, util


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Identify storm cells from radar composites and write storm polygons to the database."
    )

    parser.add_argument("starttime", help="start time (YYYYmmddHHMM)")
    parser.add_argument("endtime", help="end time (YYYYmmddHHMM)")
    parser.add_argument("config", help="configuration profile to use")

    args = parser.parse_args()

    # read configuration files
    with open(os.path.join("config", args.config, "identify_storm_cells.yaml"), "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join("config", args.config, "database.yaml"), "r") as f:
        config_database = yaml.safe_load(f)

    with open(os.path.join("config", args.config, "raster_datasources.yaml"), "r") as f:
        config_datasources = yaml.safe_load(f)

    if config["cell_identification_method"] != "contour":
        raise ValueError(
            "unknown cell identification method '{}', must be 'contour'".format(config["cell_identification_method"])
        )

    timestep = config_datasources["common"]["timestep"]

    logger = util.get_logger(config["logging"]["level"], output_file=config["logging"]["output_file"])

    database = database_methods.get_backend(config_database["database_type"])

    def worker(starttime, endtime):
        # connect to the database
        engine = database.create_engine(**config_database)
        db_conn = engine.connect()

        # delete existing database entries in the time range to avoid creating
        # duplicates
        if config["database"]["delete_existing"]:
            for quantity in config["contour_thresholds"].keys():
                for thr in config["contour_thresholds"][quantity]:
                    for area_thr in config["criteria"]["min_area"]:
                        for cluster_distance in config["clustering"]["max_cell_distance"]:
                            database.delete_all_rows(
                                starttime,
                                endtime,
                                db_conn,
                                (
                                    f"{config['contour_method_name']}_{quantity}_{thr}"
                                    f":minArea_{area_thr}"
                                    f":clusters_{cluster_distance}"
                                ),
                            )

        curtime = starttime
        storm_cells = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        n_db_entries = 0

        while curtime <= endtime:
            # go through the specified quantities: read the input rasters
            # and identify storm cells
            for quantity in config["contour_thresholds"].keys():

                for threshold, min_area in product(
                    config["contour_thresholds"][quantity], config["criteria"]["min_area"]
                ):
                    cell_method = f"{config['contour_method_name']}_{quantity}_{threshold}:minArea_{min_area}"

                    starttime = time.time()

                    try:
                        # read the input raster, projection and transformation
                        input_field, input_proj, affine = raster_datasources.get_data(
                            curtime, **config_datasources[quantity]
                        )
                        trans = Transformer.from_crs(input_proj, config_database["projection"])

                        # identify the cells
                        contour_extractor = contour.get_method(config["contour_method"])
                        contour_coords = contour_extractor(
                            input_field,
                            threshold,
                            **config["contour_extractor_kwargs"],
                        )
                        storm_polys_cur = []
                        for cc in contour_coords:
                            if len(cc) >= 3:
                                polys = _get_shapely_polygons(
                                    cc[:, 1],
                                    cc[:, 0],
                                    affine,
                                    geotrans=trans,
                                    required_orientation=config["required_orientation"],
                                    min_area=min_area,
                                    tolerance=config["simplification"]["tolerance"],
                                    preserve_topology=bool(config["simplification"]["preserve_topology"]),
                                    cluster_max_cell_dist=config["clustering"]["max_cell_distance"],
                                )
                                for poly in polys:
                                    storm_polys_cur.append(poly)

                        num_cells = len(storm_polys_cur)

                        if num_cells == 0:
                            if config["parallelization"]["num_workers"] == 1:
                                logger.info(
                                    f"no storm cells at {curtime} from {quantity}"
                                    f" / {threshold} (min area {min_area})"
                                )
                            continue

                        if config["parallelization"]["num_workers"] == 1:
                            logger.info(
                                f"identified {num_cells:d} storm cells at {curtime} "
                                f"from {quantity} / {threshold} (min area {min_area}) "
                                f"in {time.time() - starttime:.2f} seconds"
                            )

                        for cluster_distance in config["clustering"]["max_cell_distance"]:

                            cell_method = (
                                f"{config['contour_method_name']}_{quantity}_{threshold}"
                                f":minArea_{min_area}"
                                f":clusters_{cluster_distance}"
                            )
                            storm_polys_clustered, clustering_time = _apply_clustering(
                                storm_polys_cur,
                                cluster_distance,
                                config["parallelization"]["num_workers"],
                            )
                            num_clusters = len(storm_polys_clustered)

                            storm_cells[curtime][cell_method] = storm_polys_clustered
                            n_db_entries += len(storm_polys_clustered)

                            if config["parallelization"]["num_workers"] == 1:
                                if num_cells != num_clusters:
                                    logger.info(
                                        f"grouped {num_cells} storm cells into "
                                        f"{num_clusters} clusters with distance limit {cluster_distance} "
                                        f"in {clustering_time:.2f} seconds"
                                    )
                    except FileNotFoundError as e:
                        if config["parallelization"]["num_workers"] == 1:
                            logger.error(e)
                    except ModuleNotFoundError as e:
                        if config["parallelization"]["num_workers"] == 1:
                            logger.error(e)
                # except Exception as e:
                #     if config["parallelization"]["num_workers"] == 1:
                #         logger.error("unspecified error: " + str(curtime))
                #         logger.debug(traceback.print_tb(e.__traceback__))

            # write the cells into database if the buffer is full or if we are
            # at the endpoint of the time interval
            if n_db_entries >= config["database"]["commit_size"] or curtime == endtime:
                if config["parallelization"]["num_workers"] == 1:
                    starttime = time.time()

                _write_cells_to_database(
                    storm_cells,
                    db_conn,
                    database,
                    config_database,
                    config["contour_thresholds"],
                    config["contour_method_name"],
                )

                if config["parallelization"]["num_workers"] == 1:
                    logger.info(
                        "inserted {} entries to database in {:.2f} seconds".format(
                            n_db_entries, time.time() - starttime
                        ),
                    )

                storm_cells = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                n_db_entries = 0

            curtime = curtime + timedelta(minutes=timestep)

        # close the database connection
        db_conn.close()
        engine.dispose()

    starttime = datetime.strptime(args.starttime, "%Y%m%d%H%M")
    endtime = datetime.strptime(args.endtime, "%Y%m%d%H%M")

    util.process_chunked_time_range(
        starttime,
        endtime,
        timestep,
        worker,
        num_workers=config["parallelization"]["num_workers"],
        max_chunk_len=config["parallelization"]["max_chunk_len"],
    )


def _get_shapely_polygons(
    contour_x,
    contour_y,
    affine,
    geotrans=None,
    required_orientation=None,
    min_area=0.0,
    tolerance=0.0,
    preserve_topology=True,
    cluster_max_cell_dist=0.0,
):
    contour_x = contour_x + 0.5
    contour_y = contour_y + 0.5

    contour_x = affine[0, 0] + affine[0, 1] * contour_x + affine[0, 2] * contour_y
    contour_y = affine[1, 0] + affine[1, 1] * contour_x + affine[1, 2] * contour_y
    poly = shapely.geometry.Polygon(np.column_stack([contour_x, contour_y]))

    if not poly.is_valid:
        poly = poly.buffer(0)

    polys = []

    def get_polygon(poly_cand):
        if not poly_cand.is_valid or len(poly_cand.exterior.coords) == 0:
            return None

        # Orient the polygon
        poly_cand = shapely.geometry.polygon.orient(poly_cand, 1.0)
        if required_orientation in ["ccw", "cw"]:
            ring = shapely.geometry.LinearRing(poly_cand.exterior.coords[:])
            if required_orientation == "ccw" and not ring.is_ccw:
                return None
            elif required_orientation == "cw" and ring.is_ccw:
                return None
        if min_area > 0.0 and poly_cand.area / 1e6 < min_area:
            return None

        if tolerance > 0.0:
            return poly_cand.simplify(tolerance, preserve_topology=preserve_topology)
        else:
            return poly_cand

    if isinstance(poly, shapely.geometry.multipolygon.MultiPolygon):
        for poly_cur in poly.geoms:
            poly_out = get_polygon(poly_cur)
            if poly_out is not None:
                polys.append(poly_out)
    else:
        poly_out = get_polygon(poly)
        if poly_out is not None:
            polys.append(poly_out)

    if geotrans is not None:
        for i in range(len(polys)):
            polys[i] = transform(geotrans.transform, polys[i])

    return polys


# apply clustering to the storm polygons and convert to MultiPolygon
def _apply_clustering(polys, max_cell_dist, num_workers):
    clustered_polys = []

    if max_cell_dist > 0.0:
        starttime = time.time()
        clustered_polys = distance_clustering(polys, max_dist=max_cell_dist * 1000.0)
        clustering_time = time.time() - starttime
    else:
        clustered_polys = [shapely.geometry.MultiPolygon([p]) for p in polys]
        clustering_time = 0

    return clustered_polys, clustering_time


# write storm cells to database
def _write_cells_to_database(storm_cell_dict, db_conn, database_methods, config_database, contour_thrs, contour_method):
    cell_table_values = []
    attr_table_values = []

    for analysis_time in storm_cell_dict.keys():
        # for quantity in storm_cell_dict[analysis_time].keys():
        #     for threshold in storm_cell_dict[analysis_time][quantity].keys():
        #         for cluster_distance in storm_cell_dict[analysis_time][quantity][threshold].keys():
        for cell_method in storm_cell_dict[analysis_time].keys():
            for cell_idx, cell in enumerate(storm_cell_dict[analysis_time][cell_method]):
                srid = config_database["projection"] if isinstance(config_database["projection"], int) else -1
                geom = geoalchemy2.shape.from_shape(cell, srid=srid)
                cell_table_values.append(
                    {
                        "timestamp": analysis_time,
                        "identifier": cell_idx + 1,
                        "method": cell_method,
                        "geometry": geom,
                    }
                )
                vals = {
                    "timestamp": analysis_time,
                    "identifier": cell_idx + 1,
                    "method": cell_method,
                }
                vals["area"] = cell.area / 1e6
                vals["centroid_x"] = cell.centroid.x
                vals["centroid_y"] = cell.centroid.y
                attr_table_values.append(vals)

    if len(cell_table_values) > 0:
        insert = database_methods.table_stormcells.insert().values(cell_table_values)
        db_conn.execute(insert)

        insert = database_methods.table_cell_attributes.insert().values(attr_table_values)
        db_conn.execute(insert)

        db_conn.commit()


if __name__ == "__main__":
    main()
