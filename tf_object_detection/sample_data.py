"""
Scripts to resize, and/or crop labels for creating tfrecords

Allows complete 2D rectanglular AABB or objects split into sides
"""
# NOTE The cropping/resizing parts could be moved to be independent
#      of the tfrecords file creation

import hashlib

import cv2
from object_detection.utils import dataset_util
import tensorflow as tf


def clip_float(value):
    """Clips value to a range of [0,1]"""
    return min(1, max(value, 0))


def _rear_extrema(rear):
    """ x, y, width, height to x_min, x_max, y_min, y_max """
    rear_xmin = rear[0]
    rear_xmax = rear[0] + rear[2]
    rear_ymin = rear[1]
    rear_ymax = rear[1] + rear[3]
    return [rear_xmin, rear_xmax, rear_ymin, rear_ymax]


def _side_extrema(side):
    """ Sort side values to side_xmin, side_xmax, side_ymin, side_ymax """
    side_xmin = min(side[::2])
    side_xmax = max(side[0::2])
    side_ymin = min(side[1::2])
    side_ymax = max(side[1::2])
    return [side_xmin, side_xmax, side_ymin, side_ymax]


def _convert_divided_sides(rear, side):
    """ Converts labels into x_min, x_max, y_min, y_max
    Divided, as in rears and sides are stored separately"""
    classes = []
    coordinate_lists = []

    if rear is not None:  # [xmin, ymin, width, height]
        coordinate_lists.append(_rear_extrema(rear))
        classes.append('rear')

    if side is not None:
        coordinate_lists.append(_side_extrema(side))
        classes.append('side')

    return classes, coordinate_lists


def _convert_complete_sides(rear, side):
    """ Creates exactly one AABB for each object """
    classes = ['complete']

    if side is not None and rear is not None:  # combine for complete
        rear_xmin, rear_xmax, rear_ymin, rear_ymax = _rear_extrema(rear)
        side_xmin, side_xmax, side_ymin, side_ymax = _side_extrema(side)
        box_xmin = min(rear_xmin, side_xmin)
        box_xmax = max(rear_xmax, side_xmax)
        box_ymin = min(rear_ymin, side_ymin)
        box_ymax = max(rear_ymax, side_ymax)
        coordinate_lists = [[box_xmin, box_xmax, box_ymin, box_ymax]]

    elif side is not None:
        coordinate_lists = [_side_extrema(side)]

    elif rear is not None:
        coordinate_lists = [_rear_extrema(rear)]

    return classes, coordinate_lists


class SampleData:
    """ Container to modify labels for cropping and resizing """

    def __init__(self, image, key, config, crop_number, divided=False):
        self.image = image
        self.image_id = key
        self.image_crop_id = key + str(crop_number)
        self.image_path = key
        self.divided = divided
        self.label_map = {'rear': 1, 'side': 2} if divided else {'complete': 1}

        self.out_width = config['crop_output_width']
        self.out_height = config['crop_output_height']
        self.in_width = config['crop_input_width']
        self.in_height = config['crop_input_height']
        # labels are annotated relative to image size, minimum size is scaled to that
        self.min_box_width = float(config['min_width']) / float(self.out_width)
        self.min_box_height = float(config['min_height']) / float(self.out_height)
        self.image_format = config['image_format']

        self.input_crop_min_x = None
        self.input_crop_max_x = None
        self.input_crop_min_y = None
        self.input_crop_max_y = None

        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.classes = []

        # Added information for 3d-like crops
        self.num_vehicle_sides = []  # i.e. 2 for rear and side, 1 for rear, 1 for side
        self.complete_polygons = []

    def _coordinate_value_lists(self):
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def map_x_values(self, mapping):
        self.xmin = list(map(mapping, self.xmin))
        self.xmax = list(map(mapping, self.xmax))

    def map_y_values(self, mapping):
        self.ymin = list(map(mapping, self.ymin))
        self.ymax = list(map(mapping, self.ymax))

    def get_bbox_values(self, labels):
        # rear only annotation handling for complete, divided, and polygon entries
        for rear in labels[self.image_id]['boxes']:
            xmin = rear['x']
            xmax = xmin + rear['width']
            ymin = rear['y']
            ymax = ymin + rear['height']
            for container_list, value in zip(self._coordinate_value_lists(),
                                             [xmin, xmax, ymin, ymax]):
                container_list.append(value)
            self.num_vehicle_sides.append(1)
            self.classes.append('rear' if self.divided else 'complete')
            self.complete_polygons.append(rear)

        # A polygon may either contain a side, a rear, or both
        for polygon in labels[self.image_id]['polygons']:
            this_rear = polygon['rear'] if 'rear' in polygon else None
            this_side = polygon['side'] if 'side' in polygon else None

            if self.divided:
                classes, coordinate_lists = _convert_divided_sides(this_rear, this_side)
            else:
                classes, coordinate_lists = _convert_complete_sides(this_rear, this_side)
            self.classes.extend(classes)

            for coordinate_list in coordinate_lists:
                for value_list, coordinate in zip(self._coordinate_value_lists(), coordinate_list):
                    value_list.append(coordinate)

            self.complete_polygons.append(polygon)
            if this_rear and this_side:
                self.num_vehicle_sides.append(2)
            else:
                self.num_vehicle_sides.append(1)

    def transform_annotations(self, crop_window):
        """ Transforms annotations based on crop_window """
        self.input_crop_min_x = crop_window['min_x']
        self.input_crop_max_x = crop_window['max_x']
        self.input_crop_min_y = crop_window['min_y']
        self.input_crop_max_y = crop_window['max_y']

        # translate
        self.map_x_values(lambda x: x - self.input_crop_min_x)
        self.map_y_values(lambda x: x - self.input_crop_min_y)

        # scale / normalize to crop area
        self.map_x_values(lambda x: x / float(self.in_width))
        self.map_y_values(lambda x: x / float(self.in_height))

        # cut annotations to [0,1] if objects extend outside the crop
        self.map_x_values(clip_float)
        self.map_y_values(clip_float)

    def filter_too_small_annotations(self):
        """
            min_with: in pixels
            min_height: in pixels
        Notes:
            This function needs to be run! By changing resolutions, boxes may become too small
            which may lead to rounded zero-areas in tf object_detection which may in turn
            crash or silently introduce issues
        """

        remove_list = []
        for i in range(len(self.classes)):
            if (self.xmax[i] - self.xmin[i]) < self.min_box_width or\
               (self.ymax[i] - self.ymin[i]) < self.min_box_height:
                remove_list.append(i)

        for i in reversed(remove_list):
            del self.xmin[i]
            del self.xmax[i]
            del self.ymin[i]
            del self.ymax[i]
            del self.classes[i]

    def check_for_one_annotation(self):
        # returns True if at least one annotation is fully visible, returns false if not
        # NOTE This may have unintended side-effects

        def not_zero_one(value):
            # input is [0,1]
            return 0.0 < value < 1.0

        if not self.classes:  # no boxes at all
            return False
        for bbox in zip(*self._coordinate_value_lists()):
            # If none of the bounding box values are zero or one, the box is completely in the image
            # NOTE, no large cars at the border of the image may come into the tfrecord of the crop size
            if all(map(not_zero_one, bbox)):
                return True
        return False

    def basic_checks(self, ):
        # Verify length of all arrays, content is harder
        # For content checks, split function into separate parts and test indivually
        assert all(len(array) == len(self.classes) for array in self._coordinate_value_lists())
        assert all([0.0 <= value <= 1.0 for value_list in self._coordinate_value_lists() for value in value_list])

    def write_to_tfrecord(self, tfrecord_writer):
        image = cv2.resize(self.image, (self.out_width, self.out_height))
        ret_val, image = cv2.imencode('.png', image)
        image = image.tostring()
        sha256 = hashlib.sha256(image).hexdigest()
        complete_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(self.out_height),
            'image/width': dataset_util.int64_feature(self.out_width),
            'image/filename': dataset_util.bytes_feature(self.image_path.encode('utf8')),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'image/format': dataset_util.bytes_feature(self.image_format.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(self.image_crop_id.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(sha256.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(self.xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(self.xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(self.ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(self.ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(
                [x.encode('utf8') for x in self.classes]),
            'image/object/class/label': dataset_util.int64_list_feature(
                [self.label_map[x] for x in self.classes]),
        }))
        tfrecord_writer.write(complete_example.SerializeToString())
