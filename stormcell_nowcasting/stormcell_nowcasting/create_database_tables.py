"""Create tables to the database specified in config/database.yaml."""

import argparse
import os
import yaml

from stormcell_nowcasting.common import database as database_methods

parser = argparse.ArgumentParser(
    description="Create database tables.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "startyear",
    type=int,
    help="start year of the database",
)
parser.add_argument(
    "endyear",
    type=int,
    help="end year of the database",
)
parser.add_argument("config", help="configuration profile to use")
parser.add_argument("--schema", help="schema to use", default="edera")
args = parser.parse_args()

with open(os.path.join("config", args.config, "database.yaml"), "r") as f:
    config = yaml.safe_load(f)

database = database_methods.get_backend(config["database_type"])

engine = database.create_engine(**config)

srid = str(config["projection"]) if isinstance(config["projection"], int) else None
database.create_tables(engine, args.startyear, args.endyear, srid, schema=args.schema)
