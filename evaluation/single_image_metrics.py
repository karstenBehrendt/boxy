"""
Overlap metrics for comparing predictions to targets

AABB to AABB
Polygon to Polygon
"""

from boxy.evaluation.result import Result
from boxy.evaluation import polygon_iou
from boxy.common import constants
from boxy.common import helper_scripts


def __debug_view(detections, targets):
    # import here to not force more dependencies
    import cv2
    import numpy

    image = numpy.zeros((constants.HEIGHT, constants.WIDTH, 3), dtype=numpy.uint8)

    for detection in detections:
        for i in range(len(detection)):
            cv2.line(image, helper_scripts.tir(detection[i - 1]),
                     helper_scripts.tir(detection[i]), (0, 0, 255))

    for target in targets:
        for i in range(len(target)):
            cv2.line(image, helper_scripts.tir(target[i - 1]),
                     helper_scripts.tir(target[i]), (0, 255, 0))

    cv2.imshow('Detection target visualization', image)
    cv2.waitKey(1000)


def _target_to_aabb_polygons(target):
    """ Transforms target AABB into 4 point polygons
    There are AABB for rears and for complete vehicles.
    Those are moved into:
        ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))

    Parameters
    ----------
    target : dict
             single image label as provided by Boxy dataset

    Returns
        list of AABB as polygons defined by 4 points
    """
    boxes = []
    for box in target['boxes']:
        aabb = [
            (box['x'], box['y']),
            (box['x'] + box['width'], box['y']),
            (box['x'] + box['width'], box['y'] + box['height']),
            (box['x'], box['y'] + box['height'])
        ]
        boxes.append(aabb)

    for box in target['polygons']:
        box = box['bb']
        aabb = [
            (box[0], box[1]),
            (box[0] + box[2], box[1]),
            (box[0] + box[2], box[1] + box[3]),
            (box[0], box[1] + box[3])
        ]
        boxes.append(aabb)
    return boxes


def _prediction_to_aabb_polygons(prediction):
    """ Creates AABB polygons from detection boxes input

    Parameters
    ----------
    prediction: dict with 'detection_boxes'
                detection boxes is a list of AABBs
                Each box is defined as (ymin, xmin, ymax, xmax)

    Returns
        list of AABBs defined by 4 points (8 values)
    """
    boxes = []
    for box in prediction['detection_boxes']:  # [0] if directly from tf detection api
        aabb = [
            (box[1] * constants.WIDTH, box[0] * constants.HEIGHT),
            (box[3] * constants.WIDTH, box[0] * constants.HEIGHT),
            (box[3] * constants.WIDTH, box[2] * constants.HEIGHT),
            (box[1] * constants.WIDTH, box[2] * constants.HEIGHT)
        ]
        boxes.append(aabb)

    return boxes


def matching_results(detection_polygons, detection_scores, target_polygons, threshold):

    results = []
    for detection, score in zip(detection_polygons, detection_scores):
        tp = False
        for i, target in enumerate(target_polygons):
            iou = polygon_iou.polygon_iou(detection, target)
            if iou >= threshold:
                target_size = polygon_iou.polygon_size(target)
                del target_polygons[i]
                tp = True
                results.append(Result(result_type='tp', confidence=score, size=target_size))
                break
        if not tp:
            results.append(Result(result_type='fp', confidence=score,
                                  size=polygon_iou.polygon_size(detection)))

    # unmatched targets
    for target in target_polygons:
        results.append(Result(result_type='fn', size=polygon_iou.polygon_size(target)))

    return results


def match_aabb_aabb(prediction, target, threshold):
    # assumes boxes sorted by decreasing probability

    boxes = _prediction_to_aabb_polygons(prediction)
    scores = prediction['detection_scores']  # [0] if directly from tf detection api
    target_boxes = _target_to_aabb_polygons(target)

    results = matching_results(boxes, scores, target_boxes, threshold)

    return results


def evaluate_box_to_polygon(prediction, target):
    return {}


def evaluate_polygon_to_polygon(prediction, target):
    return {}


def evaluate_sides(prediction, target):
    return {}


def evaluate_rears(prediction, target):
    return {}
