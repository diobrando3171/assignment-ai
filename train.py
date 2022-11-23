import asyncio
import os
import sys
from config.base_config import BaseConfig
from utils.util import mkdir
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from models.models import create_model

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage
from data.dataprocess import get_dataloader
from models.models import save_model
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from torch.optim import lr_scheduler
import torch


basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print(basic_dir)
sys.path.append(basic_dir)


if __name__ == "__main__":
    cfg = BaseConfig().create_config()
    device = torch.device("cuda:{}".format(cfg.gpu_ids[0])) if torch.cuda.is_available() else "cpu"
    
    dir_checkpoints = os.path.join(
        basic_dir, cfg.checkpoints_dir, cfg.name)
    logger.add(os.path.join(dir_checkpoints, cfg.log_dir, "train.log"))
    #
    writer = SummaryWriter(log_dir=os.path.join(
        dir_checkpoints, cfg.log_dir))
    #
    logger.info("get data loader")
    train_loader,val_loader,train_dataset,val_dataset = get_dataloader(cfg=cfg)
    logger.info("Create model")
    model = create_model(cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    global_steps = 0
    since = time.time()
    
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            global_steps += cfg.batchSize
            # Iterate over data.
            for inputs, labels in train_loader if phase == 'train' else val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if global_steps % cfg.print_freq == 0:
                    if phase == 'train':
                        writer.add_scalar('Loss/train', running_loss, global_steps)
                    else:
                        writer.add_scalar('Loss/val', running_loss, global_steps)
                    logger.info(f"ExpName: {cfg.name} \nEpoch: {epoch}, Loss:{running_loss}, Steps: {global_steps}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n")
                   

            if phase == 'train':
                scheduler.step()
            dataset_sizes = len(train_dataset) if phase == 'train' else len(
                val_dataset)
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
           
        if epoch % 2 ==  0:
            save_info = "saving the model at the end of epoch {}, iters {}".format(
                epoch, global_steps)
            logger.info(save_info)
            save_model(cfg=cfg,model=model,epoch=epoch,save_dir=dir_checkpoints,device=device)

            logger.info(save_info)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    
    
    
    
    