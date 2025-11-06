"""Optical flow and advection-based temporal interpolation of 2d images. This
module uses the OpenCV Python interface (cv2) and optionally dask for
parallelizing the computations."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def compute_advection_field(
    image1,
    image2,
    pyr_scale=0.5,
    levels=6,
    winsize=30,
    iterations=10,
    poly_n=7,
    poly_sigma=1.5,
    filter_stddev=1.0,
    minval=None,
):
    """Compute advection field between two images by using the OpenCV Farneback method."""
    images_filtered = _filtered_ubyte_images(
        np.stack([image1, image2]), filter_stddev, minval=minval
    )
    return cv2.calcOpticalFlowFarneback(
        images_filtered[0],
        images_filtered[1],
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        0,
    )


def _filtered_ubyte_images(images, filter_stddev, minval=None):
    images = images.copy()

    if filter_stddev > 0.0:
        for i in range(images.shape[0]):
            images[i, :] = gaussian_filter(images[i, :], filter_stddev)

    if minval is None:
        minval = np.nanmin(images)
    maxval = np.nanmax(images)

    mask = images > minval

    images[mask] = (images[mask] - minval) / (maxval - minval) * 255.0
    images[~mask] = 0.0

    return images.astype(np.ubyte)
