"""Checks for issues in label files

Whenever there is an alleged issue in the label files,
I try to check if it is reproducible. I hope to add some
checks here over time.

Think of this as a growing unit test for data.
"""

import argparse
import json

import tqdm

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
            if 'AABB' not in vehicle:
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

    check_missing_aabb(labels)  # in-place manipulation!


def parse_args():
    """ Reads command line arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', help='Input boxy label file', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    check_labels(args.input_json)
