import pandas as pd
import os
import numpy as np
import random
import pickle
from scipy.ndimage import binary_fill_holes
import nibabel as nib
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.spatial_transformations import augment_spatial
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from medpy.metric import binary
from dann_utils import vendor_to_label

class DANNDataLoader:
    def __init__(self, root_dir, batch_size, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.patient_ids = [file.split(".npy")[0] for file in os.listdir(root_dir)]
        self.batch_size = batch_size
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(
                torch.utils.data.DataLoader(
                    DANNPatient(
                        root_dir, id, transform=transform, transform_gt=transform_gt
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            )
        self.counter_id = 0

    def __iter__(self):
        self.counter_iter = 0
        return self

    def set_transform(self, transform):
        for loader in self.patient_loaders:
            loader.dataset.transform = transform

    def __next__(self):
        if self.counter_iter == len(self):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if self.counter_id % len(self) == 0:
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]


class DANNPatient(torch.utils.data.Dataset):
    def __init__(self, root_dir, patient_id, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.id = patient_id
        with open("dann_preprocessed/patient_info.pkl", "rb") as f:
            self.info = pickle.load(f)[patient_id]
        self.data = np.load(
            os.path.join(self.root_dir, f"{self.id}.npy"), allow_pickle=True
        ).item()
        self.transform = transform
        self.transform_gt = transform_gt
        self.vendor = [0, 0, 0]
        self.vendor[vendor_to_label[self.info.get("VENDOR")]] = 1

    def __len__(self):
        return self.info["shape_ED"][2] + self.info["shape_ES"][2]

    def __getitem__(self, slice_id):
        is_es = slice_id >= len(self) // 2
        slice_id = slice_id - len(self) // 2 if is_es else slice_id
        sample = {
            "data": (
                self.data["data"]["ED"][slice_id]
                if not is_es
                else self.data["data"]["ES"][slice_id]
            )
        }
        if self.transform_gt:
            if self.data["gt"] != {}:
                sample["gt"] = (
                    self.data["gt"]["ED"][slice_id]
                    if not is_es
                    else self.data["gt"]["ES"][slice_id]
                )
            if self.transform:
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)
            if self.data["gt"] != {}:
                sample["gt"] = (
                    self.data["gt"]["ED"][:, :, slice_id]
                    if not is_es
                    else self.data["gt"]["ES"][:, :, slice_id]
                )
        sample['vendor'] = self.vendor
        return sample


class DANNAllPatients(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, transform_gt=True):
        self.dataloader = DANNDataLoader(
            root_dir, 1, transform=transform, transform_gt=transform_gt
        )
        self.map = [
            j for i, patient in enumerate(self.dataloader) for j in [i] * len(patient)
        ]

    def __len__(self):
        return sum(len(patient) for patient in self.dataloader)

    def __getitem__(self, id):
        patient_id = self.map[id]
        slice_id = id - self.map.index(patient_id)
        sample = self.dataloader.patient_loaders[patient_id].dataset.__getitem__(slice_id)
        return sample


class DANNPatientPhase(torch.utils.data.Dataset):
    def __init__(self, root_dir, patient_id, phase, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.id = patient_id
        with open("dann_preprocessed/patient_info.pkl", "rb") as f:
            self.info = pickle.load(f)[patient_id]
        self.data = np.load(
            os.path.join(self.root_dir, f"{self.id}.npy"), allow_pickle=True
        ).item()
        self.phase = phase
        self.transform = transform
        self.transform_gt = transform_gt

    def __len__(self):
        return self.info[f"shape_{self.phase}"][2]

    def __getitem__(self, slice_id):
        sample = {"data": self.data["data"][self.phase][slice_id]}
        if self.transform_gt:
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"][self.phase][slice_id]
            if self.transform:
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"][self.phase][:, :, slice_id]
        return sample