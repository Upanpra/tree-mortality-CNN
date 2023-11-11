import os
import numpy as np
import rasterio
import torch
from torch import nn
from typing import Optional
from tqdm import tqdm

SUPPORTED_LR_SCHEDULERS = {"cosine"}


def predictCNN(net, train_loader, output_folder: str):
    net.eval()

    # print(train_loader)
    for i, sample in tqdm(enumerate(train_loader), total= len(train_loader)):  # Loop over each batch in train_loader
        # If you are using a GPU, speed up computation by moving values to the GPU
        images = sample['image']
        if torch.cuda.is_available():
            net = net.cuda()
            images = images.cuda()

        with torch.no_grad():
            outputs = net(images)  # Forward pass: compute the output class logits given an image

        predictions = torch.argmax(outputs, dim=1)

        # loop over output batch and write each element out:
        for y_hat, filename, full_input_path in zip(predictions, sample['filename'], sample['full_input_path']):
            with rasterio.open(full_input_path) as src:
                output_raster = src.read(1)
                output_raster[:] = int(y_hat)
                filename, ext = os.path.splitext(filename)
                filename = f"{filename}_class_prediction_{int(y_hat)}{ext}"
                output_raster = output_raster.astype(np.int8)
                dst_path = os.path.join(output_folder, filename)
                kwargs = src.meta
                kwargs.update(
                    dtype=rasterio.int8,
                    count=1,
                    nodata=None
                )

            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                dst.write_band(1, output_raster)
