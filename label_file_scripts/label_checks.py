"""Checks for issues in label files

Whenever there is an alleged issue in the label files,
I try to check if it is reproducible. I hope to add some
checks here over time.

Think of this as a growing unit test for data.
"""

import argparse
import json

from boxy.common import constants


def check_out_of_bounds(labels):
    """Checks for out of bound annotations (partially outside of image)

    Parameters
    ----------
    labels : dict
             loaded content of annotation file

    Note
    ----
    Was fixed August 20, 2019
    """
    vehicle_counter = 0
    for image_label in labels.values():
        for i, vehicle in enumerate(image_label['vehicles']):
            # Not using min/max to be able to count occurances
            oob = False

            # # AABB
            if vehicle['AABB']['x1'] < 0:
                oob = True
            if vehicle['AABB']['y1'] < 0:
                oob = True
            if vehicle['AABB']['x2'] > constants.WIDTH:
                oob = True
            if vehicle['AABB']['y2'] > constants.HEIGHT:
                oob = True

            # # rear
            if vehicle['rear'] is not None:
                if vehicle['rear']['x1'] < 0:
                    oob = True
                if vehicle['rear']['y1'] < 0:
                    oob = True
                if vehicle['rear']['x2'] > constants.WIDTH:
                    oob = True
                if vehicle['rear']['y2'] > constants.HEIGHT:
                    oob = True

            # # side
            if vehicle['side'] is not None:
                for point in vehicle['side'].keys():
                    if vehicle['side'][point]['x'] < 0:
                        oob = True
                    if vehicle['side'][point]['y'] < 0:
                        oob = True
                    if vehicle['side'][point]['x'] > constants.WIDTH:
                        oob = True
                    if vehicle['side'][point]['y'] > constants.HEIGHT:
                        oob = True
            if oob:
                vehicle_counter += 1
    print('Found', vehicle_counter, 'annotations that are out of image bounds')


def check_missing_aabb(labels):
    """Checks for missing AABB in all labels

    Parameters
    ----------
    labels : dict
             loaded content of annotation file

    Note
    ----
    Was reported, could not verify.
    """
    counter = 0
    for key, label in labels.items():
        for vehicle in label['vehicles']:
            if 'AABB' not in vehicle or vehicle['AABB'] is None:
                counter += 1

    print('Found', counter, 'instances of missing AABB')


def check_labels(input_labels):
    """ Reads a label file, filters annotations by size, and creates
    a new one

    Parameters
    ----------
    input_labels : str
                   path to existing Boxy annotation file
    """
    with open(input_labels, 'r') as input_handle:
        labels = json.load(input_handle)

    check_missing_aabb(labels)
    check_out_of_bounds(labels)


def parse_args():
    """ Reads command line arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', help='Input boxy label file', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    check_labels(args.input_json)
