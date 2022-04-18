import os
import time
import torch
from config import get_parser, get_cfg
from data import prepare_dataloaders
from visualise import visualise

def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    trainloader, valloader = prepare_dataloaders(cfg)
    # print(f"Train Loader: {len(trainloader)}")
    # print(f"Val Loader: {len(valloader)}")
    for itr, data in enumerate(trainloader):
        # print(data['image'][0].shape)
        # print(data['segmentation'][0].shape)
        # print(data['instance'][0].shape)
        visualise(data, "test_image.png", img_id=itr)


if __name__ == "__main__":
    main()
