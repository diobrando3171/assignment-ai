import torch.utils.data as data
import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from torchvision import datasets
from torchvision import transforms


def loader(path):
    img = cv2.imread(path)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    gaussImg = cv2.Canny(blurred, 10, 70)

    lines = cv2.HoughLines(gaussImg, 1, np.pi/180, 120)

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    if x1 != x2 and y1 != y2:
        t = float(y2 - y1)/(x2 - x1)
    else:
        t = 0
    rotate_angle = math.degrees(math.atan(t))

    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    else:
        rotate_angle = 90 + rotate_angle

    rotated_image = ndimage.rotate(img, rotate_angle)

    return rotated_image


def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.train_image_size, cfg.train_image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root='./images/train',
                                         transform=transform,
                                         loader=loader)
    train_loader = data.DataLoader(train_dataset, batch_size=cfg.batchSize, shuffle=True,
                                   )

    validate_dataset = datasets.ImageFolder(root='./images/validate',
                                            transform=transform,
                                            loader=loader)
    validate_loader = data.DataLoader(
        validate_dataset, batch_size=cfg.batchSize, shuffle=True)

    return train_loader,validate_loader,train_dataset,validate_dataset
