"""Methods for browsing data archives."""

import os
from pathlib import Path


def get_filename(time, config):
    """Get the file name corresponding to the given time stamp, data source and configuration."""
    if "%" in config["path"]:
        path = time.strftime(config["path"])
    else:
        path = config["path"].format(year=time.year, month=time.month, day=time.day)
    if "%" in config["filename"]:
        filename = time.strftime(config["filename"])
    else:
        filename = config["filename"].format(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            minute=time.minute,
            second=time.second,
        )

    if "*" in filename:
        # Find the file in the directory with the given pattern
        files = list(Path(path).glob(filename))
        if len(files) == 0:
            raise FileNotFoundError(f"No file found in {path} with pattern {filename}")
        elif len(files) > 1:
            raise FileNotFoundError(f"Multiple files found in {path} with pattern {filename}")
        filename = str(files[0])

    return os.path.join(path, filename)
