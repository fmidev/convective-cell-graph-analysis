"""Interface module for database methods."""

from . import database_postgresql


def get_backend(backend):
    """Get database backend. The available options are 'postgresql'."""
    if backend == "postgresql":
        return database_postgresql
    else:
        raise ValueError(
            f"database backend {backend} not implemented, 'postgresql' expected"
        )
