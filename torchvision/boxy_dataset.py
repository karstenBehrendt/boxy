"""Sample torchvision object detection dataset for Boxy

Pretty much just following the pytorch tutorial at
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Some code to get started. That's all.
"""

import json
import os
import numpy as np
import torch
from PIL import Image

from boxy.common import helper_scripts
from boxy.label_file_scripts import label_reader


class BoxyDataset(object):
    def __init__(self, root, transforms, label_file):
        self.root = root
        self.transforms = transforms
        self.labels = label_reader.read_label_file(label_file)

        self.id_path_map = {}
        counter = 0
        for path, label in self.labels.items():
            if label["vehicles"]:  # vehicles in image
                self.id_path_map[counter] = path
                counter += 1

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.id_path_map[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[self.id_path_map[idx]]

        # get bounding box coordinates for each mask
        num_objs = len(label["vehicles"])
        boxes = []
        area = torch.FloatTensor([])
        for vehicle in label["vehicles"]:
            aabb = vehicle["AABB"]
            if img.size != (2464, 2056):  # Should use image constants instead
                aabb["x1"] = aabb["x1"] / 2464 * img.size[0]
                aabb["x2"] = aabb["x2"] / 2464 * img.size[0]
                aabb["y1"] = aabb["y1"] / 2056 * img.size[1]
                aabb["y2"] = aabb["y2"] / 2056 * img.size[1]

            boxes.append([aabb["x1"], aabb["y1"], aabb["x2"], aabb["y2"]])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        labels = torch.ones((num_objs,), dtype=torch.int64)  # only one class
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # to be used if zero

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = None
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.id_path_map)
