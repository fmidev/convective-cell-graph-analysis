from typing import Any, Optional

from sqlalchemy import ARRAY, Boolean, DateTime, Double, Index, Integer, PrimaryKeyConstraint, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column
from sqlalchemy.sql.sqltypes import NullType
from geoalchemy2 import Geometry
import datetime
import os

DB_USER = os.environ.get("DB_USER", None)
DB_PASSWD = os.environ.get("DB_PASSWD", None)

SCHEMA = "raincells"


def get_engine(db_config, echo=False, pool_size=5, max_overflow=10, **kwargs):
    """Create a SQLAlchemy engine from a database configuration dictionary.

    Parameters:
        db_config (dict): A dictionary containing the database configuration.
        **kwargs: Additional keyword arguments to pass to the create_engine function.

    """
    if db_config.get("user"):
        pass
    elif DB_USER is not None:
        db_config["user"] = DB_USER
    else:
        raise ValueError("No database user provided in config or environment as DB_USER.")

    # Get password from config or environment; if neither exist raise an error
    if db_config.get("password"):
        pass
    elif DB_PASSWD is not None:
        db_config["password"] = DB_PASSWD
    else:
        raise ValueError("No database password provided in config or environment as DB_PASSWD.")
    conn_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    return create_engine(conn_url, echo=echo, pool_size=pool_size, max_overflow=max_overflow)


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class NextCells(Base):
    __tablename__ = "next_cells"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "identifier", "method", "next_identifier", name="next_cells_pkey"),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    next_identifier: Mapped[int] = mapped_column(Integer, primary_key=True)


class StormcellAdditionalAttributes(Base):
    __tablename__ = "stormcell_additional_attributes"
    __table_args__ = (
        PrimaryKeyConstraint(
            "timestamp",
            "identifier",
            "method",
            "attribute",
            "attribute_type",
            name="stormcell_additional_attributes_pkey",
        ),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    attribute: Mapped[str] = mapped_column(String, primary_key=True)
    attribute_type: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[float]] = mapped_column(Double(53))


class StormcellApproxEllipse(Base):
    __tablename__ = "stormcell_approx_ellipse"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "identifier", "method", "ellipse_num", name="stormcell_approx_ellipse_pkey"),
        Index("idx_stormcell_approx_ellipse_centroid", "centroid"),
        Index("idx_stormcell_approx_ellipse_geometry", "geometry"),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    ellipse_num: Mapped[int] = mapped_column(Integer, primary_key=True)
    geometry: Mapped[Optional[Any]] = mapped_column(Geometry(geometry_type="POLYGON"))
    centroid: Mapped[Optional[Any]] = mapped_column(Geometry(geometry_type="POINT"))
    major_axis: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    minor_axis: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    major_radius: Mapped[Optional[float]] = mapped_column(Double(53))
    minor_radius: Mapped[Optional[float]] = mapped_column(Double(53))


class StormcellAttributes(Base):
    __tablename__ = "stormcell_attributes"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "identifier", "method", name="stormcell_attributes_pkey"),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    centroid_x: Mapped[Optional[float]] = mapped_column(Double(53))
    centroid_y: Mapped[Optional[float]] = mapped_column(Double(53))
    area: Mapped[Optional[float]] = mapped_column(Double(53))
    age: Mapped[Optional[float]] = mapped_column(Double(53))
    merged: Mapped[Optional[bool]] = mapped_column(Boolean)
    splitted: Mapped[Optional[bool]] = mapped_column(Boolean)
    vel_x: Mapped[Optional[float]] = mapped_column(Double(53))
    vel_y: Mapped[Optional[float]] = mapped_column(Double(53))


class StormcellHazardLevels(Base):
    __tablename__ = "stormcell_hazard_levels"
    __table_args__ = (
        PrimaryKeyConstraint(
            "timestamp",
            "identifier",
            "method",
            "hazard_type",
            "hazard_level_method",
            "hazard_level_method_type",
            name="stormcell_hazard_levels_pkey",
        ),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    hazard_type: Mapped[str] = mapped_column(String, primary_key=True)
    hazard_level_method: Mapped[str] = mapped_column(String, primary_key=True)
    hazard_level_method_type: Mapped[str] = mapped_column(String, primary_key=True)
    hazard_level: Mapped[float] = mapped_column(Double(53))


class StormcellRasterstats(Base):
    __tablename__ = "stormcell_rasterstats"
    __table_args__ = (
        PrimaryKeyConstraint(
            "timestamp", "identifier", "method", "quantity", "statistic", name="stormcell_rasterstats_pkey"
        ),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    quantity: Mapped[str] = mapped_column(String, primary_key=True)
    statistic: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Double(53))


class Stormcells(Base):
    __tablename__ = "stormcells"
    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "identifier", "method", name="stormcells_pkey"),
        Index("idx_stormcells_geometry", "geometry"),
        {"schema": SCHEMA},
    )

    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, primary_key=True)
    identifier: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String, primary_key=True)
    geometry: Mapped[Any] = mapped_column(Geometry(geometry_type="POLYGON"))
