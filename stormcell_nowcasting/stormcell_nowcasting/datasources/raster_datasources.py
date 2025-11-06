"""Interfaces for reading two-dimensional georeferenced rasters from different
data sources."""

from datetime import datetime, timedelta
import os

from stormcell_nowcasting.datasources import archive, importers


def get_data(time, **kwargs):
    """Read a two-dimensional raster from the given data source.

    Parameters
    ----------
    time : datetime.datetime
        The timestamp for which to read the data.
    kwargs : dict
        A dictionary defining the data source. The following key-value pairs
        are required:

        ============= ==================================================
        Key           Value
        ------------- --------------------------------------------------
        path          f-string specifying the path format,
                      see config/raster_datasources.yaml
        filename      f-string specifying the file name format,
                      see config/raster_datasources.yaml
        importer      the importer to use for reading the files, see
                      the importers module
        ============= ==================================================

        Optional keyword arguments scale, offset, nodata and subs are supplied
        to the importer, see the importers module.

    Returns
    -------
    out : dict
        A dictionary with the following key-value pairs:

        ============= ==================================================
        Key           Value
        ------------- --------------------------------------------------
        data          two-dimensional raster containing the data values
        projection    PROJ-compatible projection definition
        transform     2x3 affine transformation matrix,
                      the offset is included in the leftmost column
        ============= ==================================================
    """
    hourly = kwargs.get("hourly", False)
    if not hourly:
        return _get_archived_raster(time, **kwargs)
    else:
        return _get_archived_hourly_raster(time, **kwargs)


def _get_archived_raster(time, as_iris_cube=False, **kwargs):
    input_filename = archive.get_filename(time, kwargs)

    if os.path.exists(input_filename):
        importer = importers.get_method(kwargs["importer"])

        return importer(
            input_filename,
            scale=kwargs.get("scale", None),
            offset=kwargs.get("offset", None),
            nodata=kwargs.get("nodata", None),
            subs=kwargs.get("subs", None),
            **kwargs.get("importer_kwargs", {}),
        )
    else:
        raise FileNotFoundError(f"input file {input_filename} not found")


def _get_archived_hourly_raster(time, as_iris_cube=False, **kwargs):
    time1 = datetime(time.year, time.month, time.day, time.hour)
    time2 = datetime(time.year, time.month, time.day, time.hour) + timedelta(hours=1)

    input_filename1 = archive.get_filename(time1, kwargs)
    input_filename2 = archive.get_filename(time2, kwargs)

    if not os.path.exists(input_filename1):
        raise FileNotFoundError(f"input file {input_filename1} not found")
    if not os.path.exists(input_filename2):
        raise FileNotFoundError(f"input file {input_filename2} not found")

    importer = importers.get_method(kwargs["importer"])

    raster1, projection, transform = importer(
        input_filename1,
        scale=kwargs.get("scale", None),
        offset=kwargs.get("offset", None),
        nodata=kwargs.get("nodata", None),
        subs=kwargs.get("subs", None),
    )
    raster2 = importer(
        input_filename2,
        scale=kwargs.get("scale", None),
        offset=kwargs.get("offset", None),
        nodata=kwargs.get("nodata", None),
        subs=kwargs.get("subs", None),
    )[0]

    return _interpolate(raster1, time1, raster2, time2, time), projection, transform


def _interpolate(raster1, time1, raster2, time2, target_time):
    t = (target_time - time1).total_seconds() / (time2 - time1).total_seconds()
    return (1.0 - t) * raster1 + t * raster2
