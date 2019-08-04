#!/usr/bin/env python3
"""Freeze all stored training models for inference

Just a convenience script. Not really needed.
"""

# TODO Add more documentation if needed. Could be cleaner

import argparse
import os
import subprocess
import tqdm

from boxy.common import helper_scripts

# Set path before use!
OD_DIR = '/TODO/tensorflow/models/research/object_detection'
if OD_DIR.startswith('/TODO'):
    raise ValueError('Set object_detection file path')


def export_models_in_directory(train_directory, train_config):
    """ Exports object_detection models in a directory

    Stores _proto directories with frozen_graph_file.pb next to training steps
    """
    index_files = helper_scripts.get_files_from_folder(train_directory, '.index')
    print('Found {} models to export'.format(len(index_files)))

    for index_file in tqdm.tqdm(index_files, desc='exporting detection graphs'):
        prefix_path = index_file.replace('.index', '')
        proto_dir = index_file.replace('.index', '_proto')
        if os.path.isdir(proto_dir):
            continue  # skip already exported graphs
        if not os.path.samefile(train_directory, os.path.dirname(index_file)):
            continue  # don't recursively build new protos from copied indices
        os.makedirs(proto_dir, exist_ok=True)
        subprocess.call([
            'python3', os.path.join(OD_DIR, 'export_inference_graph.py'),
            '--input_type image_tensor',
            '--pipeline_config_path', train_config,
            '--trained_checkpoint_prefix', prefix_path,
            '--output_directory', proto_dir
        ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='Where everything is stored', required=True)
    parser.add_argument('--train_config', help='object detection config', required=True)
    pargs = parser.parse_args()
    return vars(pargs)


if __name__ == '__main__':
    args = parse_args()
    export_models_in_directory(args['train_dir'], args['train_config'])
