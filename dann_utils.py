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

#################
####CONSTANTS####
#################

BATCH_SIZE = 20

EPOCHS = 250

CKPT = "checkpoints/"
random.seed(0)
vendor_to_label = {
    "SIEMENS": 0,
    "Philips Medical Systems": 1,
    "GE MEDICAL SYSTEMS": 2
}

label_to_vendor = {
    0: "SIEMENS", 
    1: "Philips Medical Systems",
    2: "GE MEDICAL SYSTEMS"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vendor_info(path):
    vendor_info = pd.read_csv(path)
    vendor_info = pd.concat([vendor_info, vendor_info], ignore_index=True)
    vendor_info = vendor_info.sort_values(by=["SUBJECT_CODE"]).reset_index(drop=True)
    vendor_info["SUBJECT_CODE"] = [
        "{:03d}_{}".format(id, "SA" if i % 2 == 0 else "LA")
        for i, id in enumerate(vendor_info["SUBJECT_CODE"])
    ]
    vendor_info["PATH"] = vendor_info["SUBJECT_CODE"].apply(
        lambda x: f"./data/MnM2/dataset/{x[0:3]}/{x}_{{}}.nii.gz"
    )
    return vendor_info


def get_splits(vendor_info, dict_path):
    vendor_list = []
    for vendor in vendor_to_label:
        vendor_ind = (vendor_info[(vendor_info.index % 2 == 0) & (vendor_info['VENDOR'] == vendor)]['SUBJECT_CODE']).tolist()
        vendor_ind = [int(code[0:3]) for code in vendor_ind]
        vendor_ind = random.sample(vendor_ind, len(vendor_ind))
        vendor_list.append(vendor_ind)

    if not os.path.isfile(dict_path):
        splits = {"train": [], "val": [], "test": []}
        for id in range(3):
            splits["train"] += vendor_list[id][: int(len(vendor_list[id]) * 0.6)]
            splits["val"] += vendor_list[id][int(len(vendor_list[id]) * 0.6):int(len(vendor_list[id]) * 0.8)]
            splits["test"] += vendor_list[id][int(len(vendor_list[id]) * 0.8):]
        with open(dict_path, "wb") as f:
            pickle.dump(splits, f)
    else:
        with open(dict_path, "rb") as f:
            splits = pickle.load(f)
    return splits

def infer_predictions(inference_folder, test_loader, model=None, validator=None):
    if not os.path.isdir(inference_folder):
        os.makedirs(inference_folder)
    for patient in test_loader:
        patient_id = patient.dataset.id
        gt, prediction, reconstruction = [], [], []
        for iter, batch in enumerate(patient):
            batch = {
                "data": batch["data"].to(device),
                **({"gt": batch["gt"].to(device)} if "gt" in batch else {}),
            }
            if "gt" in batch:
                gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt) > 0 else batch["gt"]
            if model is not None:
                with torch.no_grad():
                    batch["prediction"] = model.forward(batch["data"])[0][0]
                prediction = (
                    torch.cat([prediction, batch["prediction"]], dim=0)
                    if len(prediction) > 0
                    else batch["prediction"]
                )
        if len(gt) != 0:
            gt = {
                "ED": gt[: len(gt) // 2].cpu().numpy(),
                "ES": gt[len(gt) // 2 :].cpu().numpy(),
            }

        if len(prediction) != 0:
            prediction = {
                "ED": prediction[: len(prediction) // 2].cpu().numpy(),
                "ES": prediction[len(prediction) // 2 :].cpu().numpy(),
            }
        else:
            prediction = validator.get_best_prediction("test", patient_id)

        if len(reconstruction) != 0:
            reconstruction = {
                "ED": reconstruction[: len(reconstruction) // 2].cpu().numpy(),
                "ES": reconstruction[len(reconstruction) // 2 :].cpu().numpy(),
            }

        for phase in ["ED", "ES"]:
            np.save(
                os.path.join(inference_folder, f"{patient_id}_{phase}.npy"),
                {
                    **({"gt": gt[phase]} if len(gt) != 0 else {}),
                    "prediction": prediction[phase],
                    **(
                        {"reconstruction": reconstruction[phase]}
                        if len(reconstruction) != 0
                        else {}
                    ),
                },
            )
    return