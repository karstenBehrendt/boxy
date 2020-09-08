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


class BoxyDataset(object):
    def __init__(self, root, transforms, label_file):
        self.root = root
        self.transforms = transforms
        with open(label_file) as lfh:
            self.labels = json.load(lfh)
        self.id_path_map = {idx: key for idx, key in enumerate(self.labels)}

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.id_path_map[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[self.id_path_map[idx]]

        if img.size != (2464, 2056):
            print("Boxes need to be scaled" + str(img.size))
            raise NotImplementedError("Boxes need to be scaled" + str(img.size))

        # get bounding box coordinates for each mask
        num_objs = len(label["vehicles"])
        boxes = []
        for vehicle in label["vehicles"]:
            aabb = vehicle["AABB"]
            boxes.append([aabb["x1"], aabb["y1"], aabb["x2"], aabb["y2"]])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        image_id = torch.tensor([idx])
        labels = torch.ones((num_objs,), dtype=torch.int64)  # only one class
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # to be used if zero

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = None
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.id_path_map)
