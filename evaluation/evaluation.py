""" Evaluation script for Boxy

Calculates a series of metrics to evaluate the different benchmarks

Format for the inference file:
    dict for each image
        detection_boxes list, each entry (ymin, xmin, ymax, xmax)
        detection_scores list, each entry between 0 and 1

# TODO inference format and challenges
"""

import argparse
from collections import defaultdict
import json
import os
import pprint
import pdb

import tqdm

from boxy.common import constants
from boxy.common import helper_scripts
from boxy.evaluation import single_image_metrics as sim


def __count_annotation_objects(result_list, min_size):
    return sum([1 for result in result_list if
                (result.type == 'tp' or result.type == 'fn') and
                result.size >= min_size])


def average_precisions(results):
    areas = {
        'all': (0, 3000**2),
        'resonable': (15, 3000**2),
        'tiny': (0, 15**2),
        'small': (15, 32**2),
        'medium': (32, 96**2),
        'large': (96, 3000**2)
    }

    detections = [result for result in results if (result.type == 'tp' or result.type == 'fp')]
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

    scores = {}
    for area_name, area_size in areas.items():
        num_targets = __count_annotation_objects(results, area_size[0]) -\
            __count_annotation_objects(results, area_size[1])
        tp = 0
        fp = 0
        step_size = 0.01
        threshold = .99
        recalls = [0]
        precisions = [1]

        for detection in detections:
            if not area_size[0] <= detection.size < area_size[1]:
                continue

            while detection.confidence < threshold and threshold >= 0:
                recall = 0 if num_targets == 0 else tp / num_targets
                recalls.append(recall)
                precision = 1 if tp + fp == 0 else tp / (tp + fp)
                precisions.append(precision)
                threshold -= step_size

            if detection.type == 'tp':
                tp += 1
            if detection.type == 'fp':
                fp += 1

        recall = 0 if num_targets == 0 else tp / num_targets
        recalls.append(recall)
        precision = 1 if tp + fp == 0 else tp / (tp + fp)
        precisions.append(precision)

        ap = 0
        for i in range(1, len(precisions)):
            ap += precisions[i - 1] * (recalls[i] - recalls[i - 1])
        scores[area_name] = ap

    pprint.pprint(scores)  # TODO remove

    return scores


def evaluate_on_set(inference_file_path, label_file_path=constants.TEST_LABEL_FILE):

    # Loading labels / targets
    if not os.path.isfile(label_file_path):
        raise IOError('No label file at ', label_file_path)
    with open(label_file_path, 'r') as lfp:
        labels = json.load(lfp)
        labels = {helper_scripts.get_label_base(key): label
                  for key, label in labels.items()}

    # Loading predictions / inference values
    if not os.path.isfile(inference_file_path):
        raise IOError('No inference file at ', inference_file_path)
    with open(inference_file_path, 'r') as lfp:
        predictions = json.load(lfp)
        predictions = {helper_scripts.get_label_base(key): prediction
                       for key, prediction in predictions.items()}

    # Setting evaluation scope
    iou_thresholds = [i * 0.05 + 0.5 for i in range(11)]
    functions = [sim.match_aabb_aabb]

    # The actual evaluation
    scores = defaultdict(dict)
    for function in functions:
        for threshold in tqdm.tqdm(iou_thresholds, desc='Thresholds'):
            results = []
            for key, value in tqdm.tqdm(labels.items(), desc='labels'):
                results.extend(function(predictions[key], value, threshold))
            scores[function.__name__][threshold] = average_precisions(results)

    pprint.pprint(scores)
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference_file', help="File with inference results", required=True)
    parser.add_argument('--label_file', help="Label file to evaluate against", required=False,
                        default=constants.TEST_LABEL_FILE)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    ARGS = parse_args()
    evaluate_on_set(inference_file_path=ARGS['inference_file'],
                    label_file_path=ARGS['label_file'])
