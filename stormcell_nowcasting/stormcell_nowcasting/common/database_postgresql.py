"""Methods for operating on the storm cell database. This implementation uses
PostgreSQL."""

from collections import defaultdict
from datetime import timedelta
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
import numpy as np
from networkx import DiGraph
from pandas import DataFrame
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, ARRAY, select
import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base

from stormcell_nowcasting.common.datastructures import StormCell

Base = declarative_base()

SCHEMA = "raincells"


class NextCellTable(Base):
    __tablename__ = "next_cells"

    timestamp = Column(DateTime, primary_key=True)
    identifier = Column(Integer, primary_key=True)
    method = Column(String, primary_key=True)
    next_identifier = Column(Integer, primary_key=True)
    overlap_area = Column(Float, nullable=True)

    __table_args__ = {
        "schema": SCHEMA,
        "postgresql_partition_by": "range(timestamp)",
    }


class StormTrackGraphTable(Base):
    __tablename__ = "track_graphs"

    timestamp = Column(DateTime, primary_key=True)
    identifier = Column(Integer, primary_key=True)
    method = Column(String, primary_key=True)
    min_timestamp = Column(DateTime, primary_key=True)
    max_timestamp = Column(DateTime, primary_key=True)
    graph_identifier = Column(Integer, primary_key=True)
    num_nodes = Column(Integer)
    num_edges = Column(Integer)

    __table_args__ = {
        "schema": SCHEMA,
        "postgresql_partition_by": "range(min_timestamp)",
    }


class StormCellTable(Base):
    __tablename__ = "stormcells"

    timestamp = Column(DateTime, primary_key=True)
    identifier = Column(Integer, primary_key=True)
    method = Column(String, primary_key=True)
    geometry = Column(Geometry("MULTIPOLYGON"), nullable=False)

    __table_args__ = {
        "schema": SCHEMA,
        "postgresql_partition_by": "range(timestamp)",
    }


class StormCellAttribTable(Base):
    __tablename__ = "stormcell_attributes"

    timestamp = Column(DateTime, primary_key=True)
    identifier = Column(Integer, primary_key=True)
    method = Column(String, primary_key=True)
    centroid_x = Column(Float)
    centroid_y = Column(Float)
    area = Column(Float)
    age = Column(Float)
    merged = Column(Boolean)
    splitted = Column(Boolean)
    vel_x = Column(Float)
    vel_y = Column(Float)

    __table_args__ = {
        "schema": SCHEMA,
        "postgresql_partition_by": "range(timestamp)",
    }


class StormCellRasterStatTable(Base):
    __tablename__ = "stormcell_rasterstats"

    timestamp = Column(DateTime, primary_key=True)
    identifier = Column(Integer, primary_key=True)
    method = Column(String, primary_key=True)
    quantity = Column(String, primary_key=True)
    statistic = Column(String, primary_key=True)
    value = Column(Float, nullable=False)

    __table_args__ = {
        "schema": SCHEMA,
        "postgresql_partition_by": "range(timestamp)",
    }


class MissingDataError(Exception):
    pass


table_cell_attributes = StormCellAttribTable.__table__
table_cell_raster_stats = StormCellRasterStatTable.__table__
table_next_cells = NextCellTable.__table__
table_stormcells = StormCellTable.__table__
table_track_graph = StormTrackGraphTable.__table__


def create_engine(host, database, user, password, **kwargs):
    # TODO: add documentation
    return sa.create_engine("postgresql://{}:{}@{}/{}".format(user, password, host, database))


def create_tables(engine, startyear, endyear, srid, schema="edera"):
    """Create database tables next_cells, stormcell_attributes,
    stormcell_rasterstats, stormcells and stormcell_approx_ellipse. The tables
    are partitioned so that one partition is created for each year.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        The database engine to use.
    startyear : int
        Start year of table partitioning.
    endyear : int
        End year of table partitioning.
    srid : int
        The spatial reference identifier (SRID) used for geometry data. If not
        specified, set to None.
    schema : str
        Schema to use. default: 'edera'
    """
    StormCellTable.__table_args__["schema"] = schema
    _create_table_partitions("next_cells", NextCellTable, startyear, endyear, schema=schema)
    _create_table_partitions("stormcells", StormCellTable, startyear, endyear, schema=schema)
    _create_table_partitions("stormcell_attributes", StormCellAttribTable, startyear, endyear, schema=schema)
    _create_table_partitions("track_graphs", StormTrackGraphTable, startyear, endyear, schema=schema)
    _create_table_partitions(
        "stormcell_rasterstats",
        StormCellRasterStatTable,
        startyear,
        endyear,
        schema=schema,
    )
    if srid is not None:
        _update_geom_srids(srid)
    Base.metadata.create_all(engine)


def delete_all_rows(starttime, endtime, db_conn, cell_method):
    """Delete rows from all tables in the given time range for the given cell
    identification method.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    db_conn : sqlalchemy.engine.base.Connection
        The database connection to use.
    cell_method : str
        The cell identification method.
    """
    for table in [
        NextCellTable,
        StormCellTable,
        StormCellAttribTable,
        StormCellRasterStatTable,
    ]:
        table_ = table.__table__
        delete = table_.delete().where(
            table_.c.timestamp >= starttime.isoformat(),
            table_.c.timestamp <= endtime.isoformat(),
            table_.c.method == cell_method,
        )
        db_conn.execute(delete)
        db_conn.commit()


def delete_rows(table, starttime, endtime, db_conn, cell_method):
    """Delete rows from the given table in the given time range.

    Parameters
    ----------
    table : sqlalchemy.sql.schema.Table
        A database table class defined in the database module.
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    db_conn : sqlalchemy.engine.base.Connection
        The database connection to use.
    cell_method : str
        The cell identification method.
    """
    table = table.__table__
    delete = table.delete().where(
        table.c.timestamp >= starttime.isoformat(),
        table.c.timestamp <= endtime.isoformat(),
        table.c.method == cell_method,
    )
    db_conn.execute(delete)
    db_conn.commit()


def query_storm_cells(starttime, endtime, db_conn, cell_method, min_area=None):
    """Query storm cells in the given time range.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    db_conn : sqlalchemy.engine.base.Connection
        The database connection to use.
    cell_method : str
        The cell identification method.
    min_area : float
        Minimum area of the storm cell, in SRID units as returned
        by ST_Area(geometry).

    Returns
    -------
    out : list
        List of storm cells of type datastructures.StormCell.
    """
    conditions = [
        table_stormcells.c.timestamp >= starttime.isoformat(),
        table_stormcells.c.timestamp <= endtime.isoformat(),
        table_stormcells.c.method == cell_method,
    ]
    if min_area is not None:
        conditions.append(func.ST_Area(table_stormcells.c.geometry) >= min_area)
    query = table_stormcells.select().where(*conditions)
    result = db_conn.execute(query)

    stormcells = []

    for row in result:
        stormcells.append(StormCell(row[0], row[1], to_shape(row[3]), row[3].srid))

    return stormcells


def query_storm_tracks(starttime, endtime, db_conn, cell_method, timestep):
    """Query storm tracks in the given time range.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    db_conn : sqlalchemy.engine.base.Connection
        The database connection to use.
    cell_method : str
        The cell identification method.
    timestep : int
        Time step to use for the storm track (minutes).

    Returns
    -------
    out : tuple
        Two-element tuple containing forward and backward storm tracks of type
        networkx.DiGraph.
    """
    forward_tracks = DiGraph()
    backward_tracks = DiGraph()

    # query cells
    cells = query_storm_cells(
        starttime,
        endtime + timedelta(minutes=timestep),
        db_conn,
        cell_method,
    )
    for cell in cells:
        forward_tracks.add_node((cell.obstime, cell.identifier))
        backward_tracks.add_node((cell.obstime, cell.identifier))

    # query cell successors
    query_columns = [
        table_next_cells.c.timestamp,
        table_next_cells.c.identifier,
        table_next_cells.c.next_identifier,
    ]
    query = sa.select(*query_columns).where(
        table_next_cells.c.timestamp >= starttime.isoformat(),
        table_next_cells.c.timestamp <= endtime.isoformat(),
    )
    result = db_conn.execute(query)

    for r in result:
        next_time = r[0] + timedelta(minutes=timestep)
        node1 = (r[0], r[1])
        node2 = (next_time, r[2])
        if not node1 in forward_tracks.nodes:
            raise KeyError(
                f"inconsistent forward connectivity information: source cell {node1[1]} at {node1[0]} does not exist"
            )
        if not node2 in forward_tracks.nodes:
            raise KeyError(
                f"inconsistent forward connectivity information: target cell {node2[1]} at {node2[0]} does not exist"
            )

        backward_tracks.add_edge(node2, node1)
        forward_tracks.add_edge(node1, node2)

    return forward_tracks, backward_tracks


# TODO: implement functions for inserting to the database


def _create_table_partitions(table, table_class, startyear, endyear, schema="edera"):
    for year in range(startyear, endyear + 1):
        cmd = """CREATE TABLE {}.{}_{} PARTITION of {}.{} FOR VALUES FROM ('{}-01-01 00:00') to ('{}-01-01 00:00')"""
        sa.event.listen(
            table_class.__table__,
            "after_create",
            sa.DDL(cmd.format(schema, table, year, schema, table, year, year + 1)),
        )


def _get_attrib_column(feature):
    table = table_cell_attributes

    if feature == "centroid_x":
        return table.c.centroid_x
    elif feature == "centroid_y":
        return table.c.centroid_y
    elif feature == "area":
        return table.c.area
    elif feature == "age":
        return table.c.age
    elif feature == "vel_x":
        return table.c.vel_x
    elif feature == "vel_y":
        return table.c.vel_y


def _update_geom_srids(srid, schema="edera"):
    cmd = "ALTER TABLE {} ALTER COLUMN geometry TYPE Geometry({}, {}) USING ST_SetSRID(geometry, {});"

    table_classes = [
        StormCellTable,
    ]
    table_names = [
        "stormcells",
    ]
    geom_types = [
        "MultiPolygon",
    ]

    for table_class, table_name, geom_type in zip(table_classes, table_names, geom_types):
        sa.event.listen(
            table_class.__table__,
            "after_create",
            sa.DDL(cmd.format(f"{schema}." + table_name, geom_type, srid, srid)),
        )
