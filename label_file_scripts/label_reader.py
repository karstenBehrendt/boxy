""" Convenience label reader

Pretty much just a quick wrapper around loading the json file
"""

import json


def read_label_file(label_path: str, min_height: float = 1.0, min_width=1.0):
    """ Reads label file and returns path: label dict
    Args:
        label_path: path to label file (json)
        min_height: minimum AABB height for filtering labels
        min_width: minimum AABB width for filtering labels

    You can't believe how noisy some human annotations are. That single pixel
    width and height filter are in there for a reason
    """
    with open(label_path) as lph:
        labels = json.load(lph)

    for key, label in labels.items():
        pop_list = []
        for vehicle_id, vehicle in enumerate(label["vehicles"]):
            aabb = vehicle["AABB"]
            if aabb["x2"] - aabb["x1"] < min_width or aabb["y2"] - aabb["y1"] < min_height:
                pop_list.append(vehicle_id)
        for pop_id in reversed(pop_list):
            del label["vehicles"][pop_id]

    return labels
