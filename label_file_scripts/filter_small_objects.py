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
    vehicle_counter = 0
    for image_label in tqdm.tqdm(labels.values(), desc='Filtering too small objects'):
        remove_vehicles = []
        for i, vehicle in enumerate(image_label['vehicles']):
            height = vehicle['AABB']['y2'] - vehicle['AABB']['y1']
            width = vehicle['AABB']['x2'] - vehicle['AABB']['x1']
            assert height > 0, 'label file corrupt, entries not ordered'
            assert width > 0, 'label file corrupt, entries not ordered'
            if height < min_side_length or width < min_side_length:
                remove_vehicles.append(i)
                vehicle_counter += 1

        for i in reversed(remove_vehicles):
            del image_label['vehicles'][i]

    print('Removed {} vehicles'.format(vehicle_counter))


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
