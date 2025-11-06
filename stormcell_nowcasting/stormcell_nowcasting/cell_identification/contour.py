"""Contour extraction methods from images. Currently this module implements
contour extraction by using the scikit-image (skimage) and Python OpenCV (cv2)
modules. The latter is adapted from SASSE."""

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from skimage.measure import find_contours

    SKIMAGE_IMPORTED = True
except ImportError:
    SKIMAGE_IMPORTED = False


def get_method(name):
    """Return a callable function for the given contour extraction method.
    The available methods are 'opencv' and 'skimage'."""
    return _funcs[name]


def opencv(
    image,
    thr,
    min_num_vert=4,
    min_poly_area=4,
    gaussfilter_stddev=0.0,
    scale=0.5,
    offset=-32,
    nodata=255,
    open_kernel_size=(3, 3),
    close_kernel_size=(12, 12),
    **kwargs,
):
    """Contour extration by using OpenCV. This implementation is adopted from
    SASSE. The values are assumed to be read from the GeoTIFFs

        /arch/radar/storage/YYYY/mm/dd/fmi/radar/iris/GeoTIFF/YYYYmmddHHMM_SUOMI250_FIN.tif

    containing reflectivity mosaics.

    Parameters
    ----------
    image : array-like
        Two-dimensional float array containing the input image.
    thr : float
        Intensity threshold value for the extracted contour levels.
    min_num_vert : int
        Minimum number of polygon vertices for a contour object to be valid.
    min_poly_area : int
        Minimum polygon area for a contour object to be valid.
    gaussfilter_stddev : float
        If set to nonzero value, a Gaussian filter with this standard deviation
        is applied to the input image before contour extraction.
    """
    if not CV2_IMPORTED:
        raise ModuleNotFoundError("module cv2 not found")

    image = image.copy()

    thresholded_image = (image >= thr).astype(np.uint8) * 255

    open_kernel = np.ones(open_kernel_size, np.uint8)
    close_kernel = np.ones(close_kernel_size, np.uint8)

    thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, close_kernel)
    thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, open_kernel)

    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = []
    for c in contours:
        c_ = np.empty((c.shape[0], 2), dtype=int)
        c_[:, 0] = c[:, 0, 1]
        c_[:, 1] = c[:, 0, 0]
        out.append(c_)

    return out


def skimage(
    image,
    thr,
    min_num_vert=4,
    binarize=False,
    gaussfilter_stddev=0.0,
    **kwargs,
):
    """Contour extraction by using skimage.

    Parameters
    ----------
    image : array-like
        Two-dimensional float array containing the input image.
    thr : float
        Intensity threshold value for the extracted contour levels.
    min_num_vert : int
        Minimum number of polygon vertices for a contour object to be valid.
    """
    if not SKIMAGE_IMPORTED:
        raise ModuleNotFoundError("module skimage not found")

    image = image.copy()
    image[~np.isfinite(image)] = np.nanmin(image)

    if gaussfilter_stddev > 0:
        image = gaussian_filter(image, gaussfilter_stddev)

    contours = find_contours(image, thr, positive_orientation="low")

    return contours


_funcs = {"opencv": opencv, "skimage": skimage}
