#!/usr/bin/env python3
""" Creates tf_records files for the tensorflow/object detection module for the Boxy dataset

Note that test labels are not public and the training and validation sets can be changed
however they may be convenient. This script currently can create tfrecords for
different purposes.

Please see the repo for sample configs and adapt them accordingly.
The json file contains the annotations, an image directory needs to be given for the input images.
Those may be the raw images, equalized ones or compressed ones for faster (down-)loading.

The script can easily run a day depending on the given image resolutions.

Sample usage:
    python to_tfrecords.py --create --config path_to_some_config_file
"""

import argparse
import json
import os
from random import shuffle, randint

import cv2
import tensorflow as tf
import tqdm
import yaml

from boxy.tf_object_detection.sample_data import SampleData
from boxy.common import constants


def create_tf_object_detection_tfrecords(
        labels, tfrecord_file, config, max_samples=0):
    """ Creates a tfrecord dataset split specific to tf's objection_detection module
    The created dataset is stored to file and may take up to a TB of space.
    Complete in this context means that bounding boxes fully capture each object.
    The divided files split sides and rears into separate boxes.
    All tfrecords only contain axis-aligned bounding boxes!

    Parameter
    ---------
    labels: dict
            contains all labels, keys are image path
    tfrecord_file: str
                   path to store 1 bounding box per car
    config: dict
            Config containing crop, resize, and image source information
    max_samples: int
                 0 for all, otherwise limited to that number, e.g. for faster validation

    Returns
    -------
    Nothing

    Notes
    -----
    See the config README.md and make sure it is configured as intended
    It specifies resolutions, cropping, and samples/crops per image
    """
    # (Incomplete) config check
    mandatory_config_params = ['crop_input_width', 'crop_output_height', 'image_format', 'image_folder']
    assert all(
        [md in config for md in mandatory_config_params]),\
        ("Needs %s mandatory_config_paramsi has %s") % \
        (str(mandatory_config_params), str(list(config.keys())))
    assert config['image_format'] in ['png', 'jpg', 'jpeg'], 'invalid image format'

    keys = list(labels.keys())
    shuffle(keys)

    with tf.python_io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        for i, key in tqdm.tqdm(enumerate(keys), total=len(labels),
                                desc='Creating a tfrecord file'):
            base_path = key
            # NOTE replaces .png with .png if image_format is png, TODO os.path.splitext
            base_path = base_path.replace('.png', '.' + config['image_format'])
            image_path = os.path.join(config['image_folder'], base_path)
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise IOError('Could not read {} \n Are the image folders'
                              'and formats correct?'.format(image_path))

            for crop_num in range(config['crops_per_image']):
                # randint includes both ends
                min_x = randint(0, constants.WIDTH - config['crop_input_width'])
                min_y = randint(0, constants.HEIGHT - config['crop_input_height'])
                crop_window = {
                    # NOTE did not check if max_crop_x and y are correct
                    'min_x': min_x,
                    'max_x': min_x + config['crop_input_width'],
                    'min_y': min_y,
                    'max_y': min_y + config['crop_input_height'],
                }

                sd = SampleData(
                    image=original_image[crop_window['min_y']:crop_window['max_y'],
                                         crop_window['min_x']:crop_window['max_x']],
                    key=key, divided=config['divided'], config=config, crop_number=crop_num)
                sd.get_bbox_values(labels)
                sd.transform_annotations(crop_window)
                sd.filter_too_small_annotations()
                sd.basic_checks()
                if sd.check_for_one_annotation():
                    sd.write_to_tfrecord(tfrecord_writer)

            if max_samples and i * config['crops_per_image'] >= max_samples:
                break


def create_datasets(tfrecords_config_path, max_valid):
    """Calls tfrecord creation for individual dataset splits
    Reads config for tfrecord creation settings, splits datasets, and creates
    the individual sets

    Parameters
    ----------
    tfrecords_config_path: str
                           yaml file with tfrecord configuration
    max_valid: int
               Max number of validation samples to write to the tfrecords file
               So that you don't need to load hundreds of GB for each validation set.
               0 for no limit

    Returns
    -------
    Nothing

    Notes
    -----
            This function may, depending on input resolution, output resolution,
            system performance, hard drive/ network speed, and filtering take
            up to a couple of days. One day is likely.
    """

    # Quick config check
    with open(tfrecords_config_path, 'r') as config_handle:
        config = yaml.load(config_handle)
    mandatory_params = ['tfrecords_folder', 'train_tfrecords', 'valid_tfrecords']
    assert all([mp in list(config.keys()) for mp in mandatory_params])

    os.makedirs(config['tfrecords_folder'], exist_ok=True)

    # NOTE This can be cleaned up and run in parallel
    with open(config['train_labels_json'], 'r') as labels_handle:
        train_labels = json.load(labels_handle)
    create_tf_object_detection_tfrecords(
        train_labels,
        os.path.join(config['tfrecords_folder'], config['train_tfrecords']),
        config)

    with open(config['valid_labels_json'], 'r') as labels_handle:
        valid_labels = json.load(labels_handle)
    create_tf_object_detection_tfrecords(
        valid_labels,
        os.path.join(config['tfrecords_folder'], config['valid_tfrecords']),
        config,
        max_valid)

    # TODO Could create tfrecords without label information, only images
    if 'test_labels_json' in config and config['test_labels_json']:
        with open(config['test_labels_json'], 'r') as labels_handle:
            test_labels = json.load(labels_handle)
        create_tf_object_detection_tfrecords(
            test_labels,
            os.path.join(config['tfrecords_folder'], config['test_tfrecords']),
            config)

    print('Done creating tfrecords')


def parse_args():
    """ Parses settings for tfrecord creation

    --create needs to explicitly be set, so you don't accidently overwrite or
    start to create a couple TB of files

    Returns arguments dict
    """
    def yaml_file(input_string):
        """Config input path validation """
        assert input_string.endswith('.yaml'), 'Config needs to be yaml file'
        assert os.path.isfile(input_string), 'Config {} does not exist'.format(input_string)
        return input_string

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=yaml_file,
                        help='path to tfrecords config yaml')
    parser.add_argument('--create', dest='create', action='store_true',
                        help='Create new tfrecords')
    parser.add_argument('--max_valid', dest='max_valid', type=int, default=0,
                        help='Reduces validation set size for faster loading of validation')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    if ARGS['create']:
        create_datasets(ARGS['config'], ARGS['max_valid'])
    else:
        print('Please use with "--create" to confirm that you want to create new tfrecords')
