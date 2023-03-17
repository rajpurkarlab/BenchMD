import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.catalog import DATASET_DICT

"""
Compute a dataset's training set per-channel mean and standard deviation for standardization purposes.
Also calculate the label distribution for the dataset's training and validation/test splits.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="/home/ubuntu/2022-spr-benchmarking/src/datasets/")
    parser.add_argument('--dataset', type=str, default="vindr")
    args = parser.parse_args()

    train_ds_kwargs = {"base_root": args.dataroot, "download": True, "train": True}
    val_ds_kwargs = {"base_root": args.dataroot, "download": True, "train": False}

    dataset = DATASET_DICT[args.dataset]
    train_dataset = dataset(**train_ds_kwargs)
    val_dataset = dataset(**val_ds_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    train_labels = []
    val_labels = []
    count = 0
    for ind, images, label in tqdm(train_loader):
        psum += images.sum(axis=[0, 2, 3])
        psum_sq += (images**2).sum(axis=[0, 2, 3])
        count += images.shape[0] * images.shape[2] * images.shape[3]
        train_labels.append(label.item())

    for ind, images, label in tqdm(val_loader):
        val_labels.append(label.item())

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print(f'train mean: {total_mean}')
    print(f'train std: {total_std}')
    train_label_freq = torch.histogram(torch.tensor(train_labels, dtype=torch.float32), bins=train_dataset.NUM_CLASSES).hist
    train_label_dist = torch.histogram(
        torch.tensor(train_labels, dtype=torch.float32), bins=train_dataset.NUM_CLASSES, density=True
    ).hist
    train_label_dist = train_label_dist / train_label_dist.sum()

    val_label_freq = torch.histogram(torch.tensor(val_labels, dtype=torch.float32), bins=train_dataset.NUM_CLASSES).hist
    val_label_dist = torch.histogram(
        torch.tensor(val_labels, dtype=torch.float32), bins=train_dataset.NUM_CLASSES, density=True
    ).hist
    val_label_dist = val_label_dist / val_label_dist.sum()

    print(f'train label frequencies ({train_dataset.NUM_CLASSES} classes): {train_label_freq}')
    print(f'train label distribution ({train_dataset.NUM_CLASSES} classes): {train_label_dist}')
    print(f'val label frequencies ({train_dataset.NUM_CLASSES} classes): {val_label_freq}')
    print(f'val label distribution ({train_dataset.NUM_CLASSES} classes): {val_label_dist}')
