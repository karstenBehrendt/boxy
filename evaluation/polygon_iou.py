#!/usr/bin/env python3
"""
A polygon to polygon IOU wrapper for shapely.
It is used to compare polygon detections to target values.
"""

import numpy
from shapely.geometry import Polygon


def _preprocess_polygon(polygon):
    """ Turn all kinds of supported inputs of into shapely Polygons
    Returns: shapely.geometry.Polygon representation for polygon
    Raises: ValueError for unsuitable inputs
    """

    # Could catch ValueErrors for unsuitable inputs
    polygon = numpy.array(polygon)

    if len(polygon.shape) == 1:
        if len(polygon) % 2:
            raise ValueError('Number of values for polygon not divisible by two.'
                             'Coordinates need an x and y coordinate: '.format(polygon))
        polygon = polygon.reshape((-1, 2))

    if not len(polygon.shape) == 2 or polygon.shape[1] != 2:
        raise ValueError('polygon of wrong dimensions. It should be of shape. '
                         'Should be: (num_points, 2). Input: {}'.format(polygon))

    polygon = Polygon(numpy.array(polygon))

    # Mainly for self-intersection
    if not polygon.is_valid:
        raise ValueError('polygon is invalid, likely self-intersecting: {}'.
                         format(polygon))

    return polygon


def polygon_size(poly):
    """ Calculates size / area of polygon

    Parameters
    ----------
    poly : iterable of 2d points
           e.g. [[1, 3], [1, 4], [2, 4], [2, 3]]

    Returns
        float, polygon size
    """
    poly = _preprocess_polygon(poly)
    return poly.area


def polygon_iou(poly1, poly2):
    """ IOU for non-intersecting polygons with a few only few checks
    Calculates the intersection over union between polygons, e.g. rectangles.
    It is meant for object detection network output evaluation (not only boxes).
    Args:
        poly1: iterable, list or numpy array of
            The iterable has to either be of even length (x, y, x, y ...)
            or be a list of points, i.e. ((x, y), (x, y), ...)
        poly2: iterable, list or numpy array
    Returns: IOU between to polygons
    Raises:
        ValueError for self-intersecting quads or non-box polygons
        ValueError for inputs of incorrect dimensions
    """
    poly1 = _preprocess_polygon(poly1)
    poly2 = _preprocess_polygon(poly2)


    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    return intersection_area / union_area
