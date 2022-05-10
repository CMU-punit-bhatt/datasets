import os
import time
import torch
from configs.config import get_parser, get_cfg
from data import prepare_dataloaders
from visualise import visualise
import cv2
from network import NormalizeInverse
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm
from nuscenes_carla_map import NUSCENES_NAME_CARLA_MAP, CARLA_CLASSES_NAME_TO_RGB
import argparse

def create_dir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

def collect_bev(rgb, seg, seg_viz):
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    trainloader, valloader = prepare_dataloaders(cfg)
    
    create_dir(rgb)
    create_dir(seg)
    create_dir(seg_viz)

    for itr, data in tqdm(enumerate(trainloader)):
        for i in range(7):
            segmentations = data["segmentation"][0, i, 0]
            segmentations = segmentations.cpu().numpy()
            segmentations = cv2.flip(segmentations, 0)
            width, height = segmentations.shape[:2]
            segmentations = segmentations[(height//2) - 55: (height//2) - 5, (width//2) - 25: (width //2) + 25]
            segmentations = cv2.resize(segmentations, (200, 200), interpolation = cv2.INTER_NEAREST)
            segmentation_cv = segmentations.copy()
            segmentations = Image.fromarray(segmentations.astype(np.uint8))
            segmentations.save(os.path.join(seg, f"segmentation_label_{itr}_{i}.png"))

            segmentation_map = np.zeros((segmentation_cv.shape[0], segmentation_cv.shape[1], 3))
            unique_labels = np.unique(segmentation_cv)
            for label in unique_labels:
              segmentation_map[segmentation_cv==label] = CARLA_CLASSES_NAME_TO_RGB[label]
            segmentation_map = Image.fromarray(segmentation_map.astype(np.uint8))
            segmentation_map.save(os.path.join(seg_viz, f"segmentation_image_{itr}_{i}.png"))

            images = data["image"][0, i, 1]
            denormalise_img = torchvision.transforms.Compose(
              (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              torchvision.transforms.ToPILImage(),))
            showimg = denormalise_img(images.cpu())
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
            showimg.save(os.path.join(rgb, f"front_image_{itr}_{i}.png"))

        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load labels.')
    parser.add_argument('--rgb', type=str, default='./rgb')
    parser.add_argument('--seg', type=str, default='./seg')
    parser.add_argument('--seg_viz', type=str, default="./seg_viz")
    args = parser.parse_args()
    
    collect_bev(args.rgb, args.seg, args.seg_viz)
