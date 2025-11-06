"""Miscellaneous utility methods."""

from datetime import timedelta
import dask
import logging
from logging import handlers
import sys

import numpy as np
from sklearn import preprocessing

try:
    from sklearn.decomposition import PCA

    SKLEARN_IMPORTED = True
except:
    SKLEARN_IMPORTED = False
try:
    from matplotlib.patches import Ellipse

    MATPLOTLIB_IMPORTED = True
except:
    MATPLOTLIB_IMPORTED = False
import shapely

_logging_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(logging_level, handler_args=None, output_file=None):
    """Create a Logger object.

    Parameters
    ----------
    logging_level : {'critical', 'error', 'warning', 'info', 'debug'}
        The logging level.
    handler_args : dict, optional
        Keyword arguments supplied to the logging handler.
    output_file : str, optional
        Name of the output file. If not None, a handler of type
        logging.handlers.RotatingFileHandler is used. If None, output is
        printed to sys.stdout.

    Returns
    -------
    out : logging.Logger
        The created Logger object.
    """
    logger = logging.getLogger()

    if output_file is None:
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        if handler_args is None:
            handler_args = {"maxBytes": 100000, "backupCount": 5}
        handler = handlers.RotatingFileHandler(filename=output_file, **handler_args)

    formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(_logging_levels[logging_level])

    return logger


def process_chunked_time_range(
    starttime,
    endtime,
    timestep,
    worker,
    *args,
    num_workers=1,
    max_chunk_len=None,
    **kwargs,
):
    """Process the given time range by using the given worker function. Split
    the range into chunks that are processed in parallel.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    timestep : int
        Time step (minutes).
    worker : function
        The worker function to apply to the time chunks. The function takes
        chunk start and end time as input arguments.
    num_workers : int, optional
        Number of parallel workers to use. Default: 1.
    max_chunk_len : int, optional
        Maximum chunk length.
    *args, **kwargs
        Optional positional and keyword arguments that are passed to the worker
        function.
    """
    time_ranges = get_chunked_time_range(starttime, endtime, timestep, num_workers, max_chunk_len=max_chunk_len)

    if num_workers == 1:
        for tr in time_ranges:
            worker(tr[0], tr[1], *args, **kwargs)
    else:
        res = []
        for tr in time_ranges:
            res.append(dask.delayed(worker)(tr[0], tr[1], *args, **kwargs))

        dask.compute(
            *res,
            num_workers=num_workers,
            scheduler="multiprocessing",
            chunksize=1,
        )


def get_chunked_time_range(starttime, endtime, timestep, num_chunks, max_chunk_len=None):
    """Split a time range into chunks of approximately the same length.

    Parameters
    ----------
    starttime : datetime.datetime
        Start time.
    endtime : datetime.datetime
        End time.
    timestep : int
        Step between consecutive times (minutes)
    num_chunks : int
        Number of chunks
    max_chunk_len : int, optional
        Maximum chunk length.

    Returns
    -------
    out : list
        List of tuples containing the start and end time of the chunks.
    """
    num_time_steps = int((endtime - starttime).total_seconds() / (timestep * 60) + 1)

    if num_chunks > num_time_steps:
        num_chunks = num_time_steps

    idx = np.array_split(np.arange(num_time_steps), num_chunks)

    if max_chunk_len is not None:
        array_lengths = [len(i) for i in idx]
        if np.any(np.array(array_lengths) > max_chunk_len):
            idx = np.arange(0, num_time_steps, max_chunk_len)
            if idx[-1] < num_time_steps - 1:
                idx = np.append(idx, [num_time_steps - 1])
            idx = zip(idx[:-1], np.array(idx[1:]) - 1)

    return [
        (
            starttime + timedelta(minutes=int(i[0]) * timestep),
            starttime + timedelta(minutes=int(i[-1]) * timestep),
        )
        for i in idx
    ]
