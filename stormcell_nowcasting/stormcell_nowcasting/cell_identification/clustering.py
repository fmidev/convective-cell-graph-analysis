"""Methods for spatial clustering of polygons."""

import numpy as np
from shapely.geometry import MultiPolygon
from sklearn.cluster import DBSCAN


def distance_clustering(polygons, max_dist):
    """Clustering of polygons based on a maximum distance criterion.

    Parameters
    ----------
    polygons : list
        Polygons of type shapely.geometry.Polygon.
    max_dist : int
        Maximum distance between polygons to be considered belonging to the
        same cluster.

    Returns
    -------
    out : list
        List of polygon clusters. The elements are of type
        shapely.geometry.MultiPolygon.
    """
    dist_matrix = _construct_distance_matrix(polygons, polygons, padding=max_dist)

    clustering = DBSCAN(eps=max_dist, min_samples=1, metric="precomputed").fit(dist_matrix)

    out = []

    # go through clusters with more than one element and create a MultiPolygon
    # for each
    for label in np.unique(clustering.labels_):
        idx = np.where(clustering.labels_ == label)[0]
        cur_cluster = []

        for i in idx:
            cur_cluster.append(polygons[i])

        out.append(MultiPolygon(cur_cluster))

    # assign a separate cluster for each isolated polygon (label -1)
    idx = np.where(clustering.labels_ == -1)[0]
    for i in idx:
        out.append(MultiPolygon([polygons[i]]))

    return out


def _construct_distance_matrix(polygons1, polygons2, padding=0):
    if len(polygons1) == 0 or len(polygons2) == 0:
        raise ValueError("one or more input set is empty")

    # construct array of neighbor candidates for each polygon
    neighbor_cands = np.ones(((len(polygons1), len(polygons2))), dtype=bool)
    bounds2 = np.array([p.bounds for p in polygons2])

    for i, p1 in enumerate(polygons1):
        neighbor_cands[i, bounds2[:, 0].T - p1.bounds[2] > padding] = False
        neighbor_cands[i, bounds2[:, 2].T - p1.bounds[0] < -padding] = False
        neighbor_cands[i, bounds2[:, 1].T - p1.bounds[3] > padding] = False
        neighbor_cands[i, bounds2[:, 3].T - p1.bounds[1] < -padding] = False

    neighbor_idx = np.where(neighbor_cands)
    dist_matrix = np.ones((len(polygons1), len(polygons2))) * 1e9

    for i, j in zip(neighbor_idx[0], neighbor_idx[1]):
        dist_matrix[i, j] = polygons1[i].distance(polygons2[j])
        dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix
