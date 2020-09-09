""" A super quick sample training script based on the torchvision object detection
tutorial. It's just to get something up and running to give others an easy entry into
using the Boxy dataset for object detections.

Pytorch does make this super easy. I was able to get the data reader and training started
within less than 3 hours without having used pytorch before. That's super nice.
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

NOTE: THIS IS A PROOF OF CONCEPT TRAINING RUN. THOSE PARAMETERS ARE NOT USEFUL!
"""

# Needs vision/references/detection from torchvision on PYTHONPATH
# TODO This needs to be changed. This is only to test the dataset reader
# i    and to get some sample up and running
from engine import train_one_epoch, evaluate
import utils
import transforms as T

import torch
from torchvision.models.detection import faster_rcnn

from boxy_dataset import BoxyDataset
from boxy.common import constants

import transforms as T


def get_transforms(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = BoxyDataset("/media/karsten/data_disk/boxy/boxy_images_raw", get_transforms(), constants.TRAIN_LABEL_FILE)
    dataset_valid = BoxyDataset("/media/karsten/data_disk/boxy/boxy_images_raw", get_transforms(), constants.VALID_LABEL_FILE)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # vehicle and background. Do I need background?
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_valid, device=device)
        torch.save(model, f"{epoch}_trained.pkl")

    print("Done training")


if __name__ == "__main__":
    main()
