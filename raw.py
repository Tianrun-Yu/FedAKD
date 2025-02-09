#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_data.py
Download and store FashionMNIST & CIFAR10 into RAW_DATA_ROOT.
"""

import os
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision import transforms

RAW_DATA_ROOT = "/home/tvy5242/EHR_fl/A_Experiment/DATA/raw"

def download_raw_datasets():
    os.makedirs(RAW_DATA_ROOT, exist_ok=True)

    # Download FashionMNIST (train & test)
    _ = FashionMNIST(
        root=RAW_DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    _ = FashionMNIST(
        root=RAW_DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Download CIFAR10 (train & test)
    _ = CIFAR10(
        root=RAW_DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    _ = CIFAR10(
        root=RAW_DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    print("Finished downloading raw FashionMNIST and CIFAR10 into", RAW_DATA_ROOT)

if __name__ == "__main__":
    download_raw_datasets()
