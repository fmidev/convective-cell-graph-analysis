"""Definitions for quantities used in the analysis and plotting of results"""

from collections import defaultdict

# Definitions for plotting functions
HISTOGRAM_LIMITS = {
    "area_km2": (0, 11000),
    "area": (0, 11000),
    "vil.mean": (1, 20),
    "vil.max": (1, 50),
    "vil.median": (1, 120),
    "rate.mean": (0, 120),
    "rate.max": (0, 120),
    "rate.median": (0, 75),
    "zdrcol_custom_filt_unique_d1.mean": (800, 3000),
    "zdrcol_custom_filt_unique_d1.median": (800, 3000),
    "zdrcol_custom_filt_unique_d1.max": (800, 3000),
    "zdrcol_custom_filt_unique_d1.sum": (0e3, 400e3),
    "et45ml.mean": (0, 8000),
    "et45ml.median": (0, 8000),
    "et45ml.max": (0, 15000),
    "et45ml.sum": (0e3, 1000e3),
    "rhohv.mean": (0.6, 1.0),
    "rhohv.median": (0.6, 1.0),
    "rhohv.max": (0.6, 1.0),
}
HISTOGRAM_AX_LIMITS = {
    "area_km2": (0, 750),
    "area": (0, 750),
    "vil.mean": (1, 12),
    "vil.max": (1, 50),
    "vil.median": (1, 20),
    "rate.mean": (0, 75),
    "rate.max": (0, 125),
    "rate.median": (0, 75),
    "zdrcol_custom_filt.mean": (800, 3000),
    "zdrcol_custom_filt.median": (800, 3000),
    "zdrcol_custom_filt.max": (800, 3000),
    "zdrcol_custom_filt.sum": (4e3, 120e3),
    "et45ml.mean": (0, 5000),
    "et45ml.median": (0, 5000),
    "et45ml.max": (0, 8000),
    "et45ml.sum": (0, 120e3),
    "rhohv.mean": (0.8, 1.0),
    "rhohv.median": (0.8, 1.0),
    "rhohv.max": (0.8, 1.0),
}

HISTOGRAM_NBINS = {
    "area_km2": 4400,
    "area": 4400,
    "vil.mean": 38,
    "vil.max": 49,
    "vil.median": 190,
    "rate.mean": 120,
    "rate.max": 60,
    "rate.median": 240,
    "zdrcol_custom_filt.mean": 11,
    "zdrcol_custom_filt.median": 11,
    "zdrcol_custom_filt.max": 11,
    "zdrcol_custom_filt.sum": 400,
    "et45ml.mean": 80,
    "et45ml.median": 80,
    "et45ml.max": 150,
    "et45ml.sum": 1000,
    "rhohv.mean": 400,
    "rhohv.median": 400,
    "rhohv.max": 400,
}
QTY_EQUIVALENTS = {
    "vil": "VIL",
    "rate": "RATE",
    "zdrcol_custom_filt": "zdrcol_custom_filt",
    "et45ml": "ET45ML",
    "rhohv": "RHOHV",
}

TITLES = {
    "area_km2": "Cell area [km$^2$]",
    "area": "Cell area [km$^2$]",
    "vil.mean": "Mean VIL [kg m$^{-2}$]",
    "vil.max": "Max VIL [kg m$^{-2}$]",
    "vil.median": "Median VIL [kg m$^{-2}$]",
    "rate.mean": "Mean rain rate [mm h$^{-1}$]",
    "rate.max": "Max rain rate [mm h$^{-1}$]",
    "rate.median": "Median rain rate [mm h$^{-1}$]",
    "rate.sum": "Volume rain rate [m$^3$ h$^{-1}$]",
    "zdrcol_custom_filt.mean": "Mean ZDR column height [km]",
    "zdrcol_custom_filt.median": "Median ZDR column height [km]",
    "zdrcol_custom_filt.max": "Max ZDR column [km]",
    "zdrcol_custom_filt.sum": "ZDR column volume [km$^3$]",
    "et45ml.mean": "Mean 45 dBZ echo top height \nabove 0° level [km]",
    "et45ml.median": "Median 45 dBZ echo top height \nabove 0° level [km]",
    "et45ml.max": "Max 45 dBZ echo top height \nabove 0° level [km]",
    "et45ml.sum": "45 dBZ echo top volume \nabove 0° level [km$^3$]",
    "rhohv.mean": r"Mean $\rho_{hv}$",
    "rhohv.median": r"Median $\rho_{hv}$",
    "rhohv.max": r"Max $\rho_{hv}$",
}

HISTOGRAM_DISCRETE = defaultdict(lambda: False)

QTY_FORMATS = defaultdict(lambda: lambda x, p: f"{x:.1f}")
QTY_FORMATS["zdrcol_custom_filt.max"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["zdrcol_custom_filt.mean"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["zdrcol_custom_filt.median"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["zdrcol_custom_filt.sum"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["et45ml.max"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["et45ml.mean"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["et45ml.median"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["et45ml.sum"] = lambda x, p: f"{x/1000:.1f}"
QTY_FORMATS["area"] = lambda x, p: f"{x:.0f}"


for subt in "_unique_d1":

    HISTOGRAM_LIMITS[f"zdrcol_custom_filt{subt}.mean"] = HISTOGRAM_LIMITS["zdrcol_custom_filt.mean"]
    HISTOGRAM_LIMITS[f"zdrcol_custom_filt{subt}.median"] = HISTOGRAM_LIMITS["zdrcol_custom_filt.median"]
    HISTOGRAM_LIMITS[f"zdrcol_custom_filt{subt}.max"] = HISTOGRAM_LIMITS["zdrcol_custom_filt.max"]
    HISTOGRAM_LIMITS[f"zdrcol_custom_filt{subt}.sum"] = HISTOGRAM_LIMITS["zdrcol_custom_filt.sum"]
    HISTOGRAM_LIMITS[f"et45ml{subt}.mean"] = HISTOGRAM_LIMITS["et45ml.mean"]
    HISTOGRAM_LIMITS[f"et45ml{subt}.median"] = HISTOGRAM_LIMITS["et45ml.median"]
    HISTOGRAM_LIMITS[f"et45ml{subt}.max"] = HISTOGRAM_LIMITS["et45ml.max"]
    HISTOGRAM_LIMITS[f"et45ml{subt}.sum"] = HISTOGRAM_LIMITS["et45ml.sum"]

    HISTOGRAM_AX_LIMITS[f"zdrcol_custom_filt{subt}.mean"] = HISTOGRAM_AX_LIMITS["zdrcol_custom_filt.mean"]
    HISTOGRAM_AX_LIMITS[f"zdrcol_custom_filt{subt}.median"] = HISTOGRAM_AX_LIMITS["zdrcol_custom_filt.median"]
    HISTOGRAM_AX_LIMITS[f"zdrcol_custom_filt{subt}.max"] = HISTOGRAM_AX_LIMITS["zdrcol_custom_filt.max"]
    HISTOGRAM_AX_LIMITS[f"zdrcol_custom_filt{subt}.sum"] = HISTOGRAM_AX_LIMITS["zdrcol_custom_filt.sum"]
    HISTOGRAM_AX_LIMITS[f"et45ml{subt}.mean"] = HISTOGRAM_AX_LIMITS["et45ml.mean"]
    HISTOGRAM_AX_LIMITS[f"et45ml{subt}.median"] = HISTOGRAM_AX_LIMITS["et45ml.median"]
    HISTOGRAM_AX_LIMITS[f"et45ml{subt}.max"] = HISTOGRAM_AX_LIMITS["et45ml.max"]
    HISTOGRAM_AX_LIMITS[f"et45ml{subt}.sum"] = HISTOGRAM_AX_LIMITS["et45ml.sum"]

    HISTOGRAM_NBINS[f"zdrcol_custom_filt{subt}.mean"] = HISTOGRAM_NBINS["zdrcol_custom_filt.mean"]
    HISTOGRAM_NBINS[f"zdrcol_custom_filt{subt}.median"] = HISTOGRAM_NBINS["zdrcol_custom_filt.median"]
    HISTOGRAM_NBINS[f"zdrcol_custom_filt{subt}.max"] = HISTOGRAM_NBINS["zdrcol_custom_filt.max"]
    HISTOGRAM_NBINS[f"zdrcol_custom_filt{subt}.sum"] = HISTOGRAM_NBINS["zdrcol_custom_filt.sum"]
    HISTOGRAM_NBINS[f"et45ml{subt}.mean"] = HISTOGRAM_NBINS["et45ml.mean"]
    HISTOGRAM_NBINS[f"et45ml{subt}.median"] = HISTOGRAM_NBINS["et45ml.median"]
    HISTOGRAM_NBINS[f"et45ml{subt}.max"] = HISTOGRAM_NBINS["et45ml.max"]
    HISTOGRAM_NBINS[f"et45ml{subt}.sum"] = HISTOGRAM_NBINS["et45ml.sum"]

    QTY_FORMATS[f"zdrcol_custom_filt{subt}.mean"] = QTY_FORMATS["zdrcol_custom_filt.mean"]
    QTY_FORMATS[f"zdrcol_custom_filt{subt}.median"] = QTY_FORMATS["zdrcol_custom_filt.median"]
    QTY_FORMATS[f"zdrcol_custom_filt{subt}.max"] = QTY_FORMATS["zdrcol_custom_filt.max"]
    QTY_FORMATS[f"zdrcol_custom_filt{subt}.sum"] = QTY_FORMATS["zdrcol_custom_filt.sum"]
    QTY_FORMATS[f"et45ml{subt}.mean"] = QTY_FORMATS["et45ml.mean"]
    QTY_FORMATS[f"et45ml{subt}.median"] = QTY_FORMATS["et45ml.median"]
    QTY_FORMATS[f"et45ml{subt}.max"] = QTY_FORMATS["et45ml.max"]
    QTY_FORMATS[f"et45ml{subt}.sum"] = QTY_FORMATS["et45ml.sum"]

    # TITLES[f"zdrcol_custom_filt{subt}.mean"] = f"Mean ZDR column height {subt} [km]"
    # TITLES[f"zdrcol_custom_filt{subt}.median"] = f"Median ZDR column height {subt} [km]"
    # TITLES[f"zdrcol_custom_filt{subt}.max"] = f"Max ZDR column {subt} [km]"
    # TITLES[f"zdrcol_custom_filt{subt}.sum"] = f"ZDR column volume {subt} [km$^3$]"

    # TITLES[f"et45ml{subt}.mean"] = f"Mean 45 dBZ echo top height {subt} \nabove 0° level [km]"
    # TITLES[f"et45ml{subt}.median"] = f"Median 45 dBZ echo top height {subt} \nabove 0° level [km]"
    # TITLES[f"et45ml{subt}.max"] = f"Max 45 dBZ echo top height {subt} \nabove 0° level [km]"
    # TITLES[f"et45ml{subt}.sum"] = f"45 dBZ echo top volume {subt} \nabove 0° level [km$^3$]"

    TITLES[f"zdrcol_custom_filt{subt}.mean"] = f"Mean ZDR column height [km]"
    TITLES[f"zdrcol_custom_filt{subt}.median"] = f"Median ZDR column height [km]"
    TITLES[f"zdrcol_custom_filt{subt}.max"] = f"Max ZDR column [km]"
    TITLES[f"zdrcol_custom_filt{subt}.sum"] = f"ZDR column volume [km$^3$]"

    TITLES[f"et45ml{subt}.mean"] = f"Mean 45 dBZ echo top height \nabove 0° level [km]"
    TITLES[f"et45ml{subt}.median"] = f"Median 45 dBZ echo top height \nabove 0° level [km]"
    TITLES[f"et45ml{subt}.max"] = f"Max 45 dBZ echo top height \nabove 0° level [km]"
    TITLES[f"et45ml{subt}.sum"] = f"45 dBZ echo top volume \nabove 0° level [km$^3$]"

    QTY_EQUIVALENTS[f"zdrcol_custom_filt{subt}"] = f"zdrcol_custom_filt{subt}"
    QTY_EQUIVALENTS[f"et45ml{subt}"] = f"et45ml{subt}"
