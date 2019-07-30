#!/usr/bin/env python3
"""
Remove need for separate polygons and boxes

Sample use:
    python restructure_labels.py
        --input_json boxy_labels_train.json
        --output_json boxy_labels_train_min_15.json
"""

import argparse
import json
import pdb

import tqdm


def restructure_all_labels(labels):
    """ No docs, will only be used once by myself """
    new_labels = {}
    counter = 0

    for key, image_label in tqdm.tqdm(labels.items(), desc='Structuring objects'):
        new_label = {}
        new_label['flaws'] = image_label['box_errors'] + image_label['cuboid_errors']
        new_label['vehicles'] = []

        for i, box in enumerate(image_label['boxes']):
            polygon = {}
            polygon['AABB'] = {}
            polygon['AABB']['x1'] = box['x']
            polygon['AABB']['x2'] = box['x'] + box['width']
            polygon['AABB']['y1'] = box['y']
            polygon['AABB']['y2'] = box['y'] + box['height']
            polygon['rear'] = polygon['AABB'].copy()
            polygon['side'] = None
            new_label['vehicles'].append(polygon)
            counter += 1

        for i, poly in enumerate(image_label['polygons']):
            polygon = {}

            if 'side' in poly and poly['side']:
                polygon['side'] = {}
                for j in range(4):
                    polygon['side']['p' + str(j)] = {}
                    polygon['side']['p' + str(j)]['x'] = poly['side'][2 * j]
                    polygon['side']['p' + str(j)]['y'] = poly['side'][2 * j + 1]
            else:
                polygon['side'] = None

            if 'rear' in poly and poly['rear']:
                polygon['rear'] = {}
                polygon['rear']['x1'] = poly['rear'][0]
                polygon['rear']['y1'] = poly['rear'][1]
                polygon['rear']['x2'] = poly['rear'][0] + poly['rear'][2]
                polygon['rear']['y2'] = poly['rear'][1] + poly['rear'][3]
            else:
                polygon['rear'] = None

            polygon['AABB'] = {}
            polygon['AABB']['x1'] = poly['bb'][0]
            polygon['AABB']['y1'] = poly['bb'][1]
            polygon['AABB']['x2'] = poly['bb'][0] + poly['bb'][2]
            polygon['AABB']['y2'] = poly['bb'][1] + poly['bb'][3]
            new_label['vehicles'].append(polygon)
            counter += 1

        new_labels[key] = new_label

    print(counter, 'vehicles')
    return new_labels


def restructure_labels(input_labels, output_labels):
    """ Creates cleaner labels

    Parameters
    ----------
    input_labels : str
                   path to existing Boxy annotation file
    output_labels : str
                    path to annotation file to be created
    """
    with open(input_labels, 'r') as input_handle:
        labels = json.load(input_handle)

    labels = restructure_all_labels(labels)

    with open(output_labels, 'w') as output_handle:
        json.dump(labels, output_handle)


def parse_args():
    """ Reads command line arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', help='Input boxy label file', required=True)
    parser.add_argument('--output_json', help='Output boxy label file', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    restructure_labels(args.input_json, args.output_json)
