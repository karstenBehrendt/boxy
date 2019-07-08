#!/usr/bin/env python3
"""
Creates new label file with minimum label size

Sample use:
    python filter_small_objects.py --input_json boxy_labels_train.json
        --output_json boxy_labels_train_min_15.json --min_side_length 15
"""
# NOTE Barely checked

import argparse
import json

import tqdm


def filter_small_objects(labels, min_side_length):
    """ Removes small objects from labels

    Parameters
    ----------
    label : dict
            as provided by Boxy
    min_side_length : scalar
                      minimum width and height for each object
    """
    box_counter = 0
    polygon_counter = 0
    for image_label in tqdm.tqdm(labels.values(), desc='Filtering too small objects'):
        remove_boxes = []
        for i, box in enumerate(image_label['boxes']):
            if box['height'] < min_side_length or box['width'] < min_side_length:
                remove_boxes.append(i)
                box_counter += 1
        for i in reversed(remove_boxes):
            del image_label['boxes'][i]

        remove_polygons = []
        for i, polygon in enumerate(image_label['polygons']):
            if polygon['bb'][2] < min_side_length or polygon['bb'][3] < min_side_length:
                remove_polygons.append(i)
                polygon_counter += 1
        for i in reversed(remove_polygons):
            del image_label['polygons'][i]

    print('Removed {} boxes and {} polygons'.format(box_counter, polygon_counter))


def filter_labels(input_labels, output_labels, min_side_length):
    """ Reads a label file, filters annotations by size, and creates
    a new one

    Parameters
    ----------
    input_labels : str
                   path to existing Boxy annotation file
    output_labels : str
                    path to annotation file to be created
    min_side_length : scalar
                      Minimum width and height per label
    """
    with open(input_labels, 'r') as input_handle:
        labels = json.load(input_handle)

    filter_small_objects(labels, min_side_length)  # NOTE in place!

    with open(output_labels, 'w') as output_handle:
        json.dump(labels, output_handle)


def parse_args():
    """ Reads command line arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', help='Input boxy label file', required=True)
    parser.add_argument('--output_json', help='Output boxy label file', required=True)
    parser.add_argument('--min_side_length', type=float, required=True,
                        help='Minimum length for each side of the AABBs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    filter_labels(args.input_json, args.output_json, args.min_side_length)
