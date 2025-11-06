import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
import os

DB_USER = os.environ.get("DB_USER", None)
DB_PASSWD = os.environ.get("DB_PASSWD", None)


def load_stormcells(db_config, timestep):
    """
    Load storm cells from the database.
    """
    # Create an engine to connect to the database

    # Get user from config or environment; if neither exist raise an error
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
    engine = create_engine(conn_url)

    # Load storm cells
    # TODO: unsafe, use parameterized queries
    query = f"""
    SELECT
        *
    FROM
        {db_config['schema']}.{db_config['table']}
    WHERE
        timestamp = '{timestep}'
    """

    stormcells = gpd.read_postgis(query, engine, geom_col="geometry")

    return stormcells
