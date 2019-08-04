#!/usr/bin/env python3
"""
Takes a frozen model and runs inference on an image folder.
The results are stored in a json file.
The input images can be stored by the images contained in a label file
"""

# TODO Add proper docstrings. Can be cleaned up

import argparse
import json

import cv2
import numpy
import tensorflow as tf
import tqdm

from boxy.common.helper_scripts import get_files_from_folder
from boxy.common.helper_scripts import get_label_base
from boxy.common.helper_scripts import tir

DEBUG = True


def load_graph(frozen_graph_filename):
    """Load frozen graph / profobuf file and return graph"""
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def jsonify(some_dict):
    """Turn numpy arrays in dict into lists for easier serialization with json"""
    assert all(isinstance(some_value, numpy.ndarray) for some_value in some_dict.values())
    return {key: value.tolist() for key, value in some_dict.items()}


def folder_bb_inference(image_folder, frozen_graph_path, label_file):
    """Run inference on all images in folder that are part of the label file"""
    print('Getting image files at', image_folder)  # Could log instead
    image_paths = get_files_from_folder(image_folder, '.png')

    if label_file is not None:
        print('Reading label file')
        with open(label_file, 'r') as lf:
            labels = list(json.load(lf).keys())
            labels = list(map(get_label_base, labels))
        print('Filtering images')
        image_paths = [image_path for image_path in image_paths
                       if get_label_base(image_path) in labels]

    graph = load_graph(frozen_graph_path)
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            detections = {}
            for index, image_path in tqdm.tqdm(enumerate(image_paths)):
                image = cv2.imread(image_path)
                output_dict = sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: numpy.expand_dims(image, 0)})
                output_dict = jsonify(output_dict)

                # "flatten" output dict
                for detection_key in ['num_detections', 'detection_boxes',
                                      'detection_scores', 'detection_classes']:
                    output_dict[detection_key] = output_dict[detection_key][0]

                detections[image_path] = output_dict

                if DEBUG:
                    for box_number, detection in enumerate(output_dict['detection_boxes']):
                        if output_dict['detection_scores'][box_number] > .1:
                            cv2.rectangle(image,
                                          tir((detection[1] * image.shape[1], detection[0] * image.shape[0])),
                                          tir((detection[3] * image.shape[1], detection[2] * image.shape[0])),
                                          (0, 0, 255),
                                          3)
                    image = cv2.resize(image, (1232, 1028))
                    cv2.imshow('image', image)
                    cv2.waitKey(1)
    return detections


def bb_inference(frozen_bb_path, image_folder, label_file, output_json):
    detections = folder_bb_inference(image_folder, frozen_bb_path, label_file)
    with open(output_json, 'w') as detections_handle:
        json.dump(detections, detections_handle)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_detector_path', help='Path to frozen bb detector', required=True)
    parser.add_argument('--image_folder', help='Image folder to process', required=True)
    parser.add_argument('--output_json', help='Results file to be created', required=True)
    # TODO requires only minimal changes to be run without label file
    parser.add_argument('--label_file', help='Filteres images by existing labels',
                        default=None, required=True)
    pargs = parser.parse_args()
    return vars(pargs)


if __name__ == '__main__':
    args = parse_args()
    bb_inference(args['frozen_detector_path'], args['image_folder'], args['label_file'], args['output_json'])
