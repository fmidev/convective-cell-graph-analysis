"""Methods for reading raster data from different sources."""

import warnings

warnings.simplefilter("ignore", UserWarning)

import h5py
from osgeo import gdal
import numpy as np
import pyproj
import contextlib
from pyproj import CRS, Transformer
from datetime import datetime
import gzip
from pathlib import Path
import rioxarray
import xarray as xr

try:
    import radlib

    RADLIB_AVAILABLE = True
except ImportError:
    RADLIB_AVAILABLE = False

PROJ4STR = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"

LL_lon = 3.169
LL_lat = 43.6301
UL_lon = 2.6896
UL_lat = 49.3767
LR_lon = 11.9566
LR_lat = 43.6201
UR_lon = 12.4634
UR_lat = 49.3654
XSIZE = 710
YSIZE = 640

# Get projection
crs = CRS.from_proj4(PROJ4STR)
crs_4326 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_4326, crs, always_xy=True)
x1, y1 = transformer.transform(
    LL_lon,
    LL_lat,
)
x2, y2 = transformer.transform(
    UR_lon,
    UR_lat,
)
UL_X, UL_Y = transformer.transform(
    UL_lon,
    UL_lat,
)
LR_X, LR_Y = transformer.transform(
    LR_lon,
    LR_lat,
)
XSCALE = (x2 - x1) / XSIZE
YSCALE = (y2 - y1) / YSIZE

XCOORDS = np.linspace(
    x1,
    x2,
    XSIZE,
)
YCOORDS = np.linspace(
    y1,
    y2,
    YSIZE,
)


def get_method(name):
    """Return the requested importer method. The currently implemented methods
    are geotiff: FMI GeoTIFF files and opera: OPERA ODIM HDF5 files."""
    return _funcs[name]


def quality_field(filename, **kwargs):
    """Read a quality field from Feldmann et al 2021
    (https://wcd.copernicus.org/articles/2/1225/2021/wcd-2-1225-2021.html)"""
    data = np.reshape(np.fromfile(filename, dtype=np.float32), [640, 710])
    data = np.flipud(data)
    transform = np.array(
        [
            UL_X,
            XSCALE,
            0,
            UL_Y,
            0,
            -YSCALE,
        ]
    ).reshape((2, 3))

    return data, PROJ4STR, transform


def netcdf(filename, variable=None, **kwargs):
    """Read data from a NetCDF file."""

    ds = xr.open_dataset(filename)
    ds = ds.rio.write_crs(ds.spatial_ref.crs_wkt)

    variables = list(ds.data_vars.keys())
    try:
        variables.remove("spatial_ref")
    except ValueError:
        pass

    if variable is None and len(variables) == 1:
        variable = variables[0]
    elif variable is None:
        raise ValueError(f"Multiple variables found in file: {variables}, but none specified in options.")

    data = ds[variable].values.squeeze()

    if kwargs.get("flipud", False):
        data = np.flipud(data)

    if "min_value" in kwargs:
        min_value = kwargs["min_value"]
        data[data < min_value] = np.nan
    if "max_value" in kwargs:
        max_value = kwargs["max_value"]
        data[data > max_value] = np.nan

    if kwargs.get("hardcode_swiss_projection", False):
        transform = np.array(
            [
                UL_X,
                XSCALE,
                0,
                UL_Y,
                0,
                -YSCALE,
            ]
        ).reshape((2, 3))
        proj4str = PROJ4STR
    else:
        transform = np.array(ds.rio.transform().to_gdal()).reshape((2, 3))
        proj4str = ds.rio.crs.to_proj4()
    return data, proj4str, transform


def zdrcol_npz(filename, **kwargs):
    """Read data from a ZDR column npz file."""

    with gzip.GzipFile(filename, "rb") as f:
        data = np.load(f, allow_pickle=True)
        data[data == 0] = np.nan

    # timestamp = datetime.strptime(
    #     Path(filename).name.split("_")[0],
    #     "%Y%m%d%H%M%S",
    # )

    transform = np.array(
        [
            UL_X,
            XSCALE,
            0,
            UL_Y,
            0,
            -YSCALE,
        ]
    ).reshape((2, 3))

    return data, PROJ4STR, transform


def metranet(filename, **kwargs):
    """Read data from MeteoSwiss Metranet format.

    Parameters
    ----------
    filename : str
        Filepath to read.


    Returns
    -------
    out : tuple
        A three-element tuple containing the read data raster, projection and
        a 2x3 matrix defining the geographical transformation. The offset is
        included in the leftmost column of the matrix.

    Raises
    ------
    ImportError
        If radblib is not available.
    NotImplementedError
        If the file is of unknown type.

    """

    if not RADLIB_AVAILABLE:
        raise ImportError(f"radlib needed to read file {filename} but not available!")

    # Read data
    with contextlib.redirect_stdout(None):
        radar_data = radlib.read_file(str(filename), physic_value=True)

    quantity = radar_data.header["product"]
    data = radar_data.data

    if "min_value" in kwargs:
        min_value = kwargs["min_value"]
        data[data < min_value] = np.nan
    if "max_value" in kwargs:
        max_value = kwargs["max_value"]
        data[data > max_value] = np.nan

    transform = np.array(
        [
            UL_X,
            XSCALE,
            0,
            UL_Y,
            0,
            -YSCALE,
        ]
    ).reshape((2, 3))

    return data, PROJ4STR, transform


def geotiff(filename, scale=None, offset=None, nodata=None, subs=None):
    """Read a GeoTIFF file.

    Parameters
    ----------
    filename : str
        The name of the file to read from.
    scale : float, optional
        The applied scale factor. If None, attempt to read from the file.
    offset : float, optional
        The applied offset. If None, attempt to read from the file.
    nodata : float, optional
        No data value. If None, attempt to read from the file.
    subs : dict, optional
        Optional dictionary of substituted values: key -> value.

    Returns
    -------
    out : tuple
        A three-element tuple containing the read data raster, projection and
        a 2x3 matrix defining the geographical transformation. The offset is
        included in the leftmost column of the matrix.
    """
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)

    projection = dataset.GetProjection()
    transform = np.array(dataset.GetGeoTransform()).reshape((2, 3))

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    scale_ = band.GetScale() if scale is None else scale
    if scale_ is None:
        scale_ = 1.0
    offset_ = band.GetOffset() if offset is None else offset
    if offset_ is None:
        offset_ = 0.0
    nodata_ = band.GetNoDataValue() if nodata is None else nodata

    data_out = np.ones(data.shape) * np.nan
    if nodata_ is not None:
        valid_mask = data != nodata_
        data_out[valid_mask] = scale_ * data[valid_mask] + offset_
    else:
        data_out = scale_ * data.astype(float) + offset_

    if subs is not None:
        for k in subs.keys():
            data_out[data_out == k] = subs[k]

    return data_out, projection, transform


def opera_odim_hdf5(filepath, scale=None, offset=None, nodata=None, subs=None):
    """Read an OPERA ODIM HDF5 file.

    Parameters
    ----------
    filename : str
        The name of the file to read from.
    scale : float, optional
        The applied scale factor. If None, attempt to read from the file.
    offset : float, optional
        The applied offset. If None, attempt to read from the file.
    nodata : float, optional
        No data value. If None, attempt to read from the file.
    subs : dict, optional
        Optional dictionary of substituted values: key -> value.

    Returns
    -------
    out : tuple
        A three-element tuple containing the read data raster, projection and
        a 2x3 matrix defining the geographical transformation. The offset is
        included in the leftmost column of the matrix.
    """
    with h5py.File(filepath, "r") as file:
        data = np.array(file["dataset1"]["data1"]["data"])
        location_data = file["where"].attrs

        def get_data_attr(name):
            try:
                dataset_what = file["dataset1"]["what"].attrs
                return dataset_what[name]
            except KeyError:
                data_what = file["dataset1"]["data1"]["what"].attrs
                return data_what[name]

        scale_ = get_data_attr("gain") if scale is None else scale
        if scale_ is None:
            scale_ = 1.0
        offset_ = get_data_attr("offset") if offset is None else offset
        if offset_ is None:
            offset_ = 0.0
        nodata_ = get_data_attr("nodata") if nodata is None else nodata

        data_out = np.ones(data.shape) * np.nan
        if nodata_ is not None:
            valid_mask = data != nodata_
            data_out[valid_mask] = scale_ * data[valid_mask] + offset_
        else:
            data_out = scale_ * data.astype(float) + offset_

        if subs is not None:
            for k in subs.keys():
                data_out[data_out == k] = subs[k]

        projection = location_data["projdef"].decode()
        pr = pyproj.Proj(projection)
        ul_x, ul_y = pr(location_data["UL_lon"], location_data["UL_lat"])
        transform = np.array(
            [
                ul_x,
                location_data["xscale"],
                0,
                ul_y,
                0,
                -location_data["yscale"],
            ]
        ).reshape((2, 3))

    return data_out, projection, transform


_funcs = {
    "geotiff": geotiff,
    "opera_odim_hdf5": opera_odim_hdf5,
    "metranet": metranet,
    "zdrcol_npz": zdrcol_npz,
    "netcdf": netcdf,
    "quality_field": quality_field,
}
