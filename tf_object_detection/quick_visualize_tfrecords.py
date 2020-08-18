#!/usr/bin/env python3
"""
Have you ever trained without bounding boxes, too small bounding boxes, incorrect classes,
too big bounding boxes, otherwise incorrect boxes? Doesn't work too well.

This script is for you. A bunch of failure cases may be directly visible.
"""

import argparse
import sys

import cv2
import tensorflow as tf


def visual_inspection(tfrecords_path, min_height=0, min_width=0, classes=None, debug_info=True):
    """ Visual inspection if values and images look reasonable
    This does not replace an actual inspection of the data, e.g. counting the samples.
    Press q to quit. Any other key to show the next sample.

    Parameters
    ----------

    tfrecords_path: str
                    path to tfrecords file with tf_object_detection api format
    min_height: float
                minimum height in pixels / image height
    min_width: float
               minimum width in pixels / image width
    classes: list
             list of available classes, e.g. [1] for only cars, [1, 2] rears and sides
    debug_info: bool
                Whether or not to print bounding box values

    Notes
    -----
    Simple script to check for first tfecord file sanity. To keep you sane.

    Throws
    ------
    IOError if tfrecords_path cannot be opened
    """

    if classes is None:
        classes = [0, 1]

    # TODO Deprecated. Should be replaced.
    record_iterator = tf.compat.v1.io.tf_record_iterator(tfrecords_path)
    with tf.compat.v1.Session():
        for record_id, record in enumerate(record_iterator):
            print(record_id)

            example = tf.train.Example()
            example.ParseFromString(record)
            image = example.features.feature['image/encoded'].bytes_list.value[0]
            height = example.features.feature['image/height'].int64_list.value[0]
            width = example.features.feature['image/width'].int64_list.value[0]
            # filename = example.features.feature['image/filename'].bytes_list.value[0]
            image_format = example.features.feature['image/format'].bytes_list.value[0]
            xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
            xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
            ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
            ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
            class_ids = example.features.feature['image/object/class/label'].int64_list.value
            class_texts = example.features.feature['image/object/class/text'].bytes_list.value[0]

            if image_format.decode('utf-8') == 'png':
                image = tf.image.decode_png(image, channels=3).eval()
            elif image_format.decode('utf-8') == 'jpeg':
                image = tf.image.decode_jpeg(image, channels=3).eval()
            else:
                raise ValueError('Imageformat %s not recognized' % image_format)

            # NOTE pre-visual checks for value ranges, if defined
            assert all([class_id in classes for class_id in class_ids])
            assert all([min_val <= max_val for min_val, max_val in zip(xmin, xmax)])
            assert all([min_val <= max_val for min_val, max_val in zip(ymin, ymax)])
            assert all([max_val - min_val >= min_width for min_val, max_val in zip(xmin, xmax)])
            assert all([max_val - min_val >= min_height for min_val, max_val in zip(ymin, ymax)])

            for box in range(len(xmin)):
                cv2.rectangle(image,
                              (int(round(xmin[box] * width)),
                               int(round(ymin[box] * height))),
                              (int(round(xmax[box] * width)),
                               int(round(ymax[box] * height))),
                              (0, 255, 0),
                              3)

            if debug_info:
                print('xmin', list(map(lambda x: x * image.shape[1], xmin)))
                print('xmax', list(map(lambda x: x * image.shape[1], xmax)))
                print('--')
                print('ymin', list(map(lambda x: x * image.shape[0], ymin)))
                print('ymax', list(map(lambda x: x * image.shape[0], ymax)))
                print('#####')
                print('classes', class_ids)
                print('classes', class_texts)
                print('#####')

            cv2.imshow('image', cv2.resize(image, (1920, 1280)))
            k = cv2.waitKey(0)
            if k == ord('q'):
                sys.exit(0)


def parse_args():
    """ Parses command line args """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tfrecord_path', type=str, help='path to tfrecord file')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    visual_inspection(ARGS['tfrecord_path'])
