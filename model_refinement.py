#!/usr/bin/env python
# coding: utf-8

# # Out-of-Distribution Detection for Model Refinement in Cardiac Image Segmentation

# In[1]:


# get_ipython().system('nvidia-smi -L')


# In[2]:


import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ## Data preparation

# In[3]:


import os
from utils import get_vendor_info
from utils import get_splits
from utils import generate_patient_info, crop_image
from utils import preprocess, preprocess_image, inSplit


vendor_info = get_vendor_info("./data/MnM2/dataset_information.csv")
vendor_info 


# In[5]:


vendor_info[['VENDOR','SUBJECT_CODE']].groupby(['VENDOR']).count().rename(columns={'SUBJECT_CODE':'NUM_STUDIES'})


# In[6]:


if not os.path.isdir("preprocessed"):
    os.makedirs("preprocessed")
splits = get_splits(os.path.join("preprocessed", "splits.pkl"))


# In[7]:


patient_info = generate_patient_info(vendor_info, os.path.join("preprocessed", "patient_info.pkl"))


# In[ ]:


spacings = [
    patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (
        splits["train"]["lab"] + splits["train"]["ulab"] + splits["val"]
    )
]
spacing_target = np.percentile(np.vstack(spacings), 50, 0)

if not os.path.isdir("preprocessed/training/labelled"): os.makedirs("preprocessed/training/labelled")
if not os.path.isdir("preprocessed/training/unlabelled"): os.makedirs("preprocessed/training/unlabelled")
if not os.path.isdir("preprocessed/validation/"): os.makedirs("preprocessed/validation/")
if not os.path.isdir("preprocessed/soft_validation/"): os.makedirs("preprocessed/soft_validation/")
if not os.path.isdir("preprocessed/testing/"): os.makedirs("preprocessed/testing/")

preprocess(
    {k:v for k,v in patient_info.items() if inSplit(k, splits["train"]["lab"])},
    spacing_target, "preprocessed/training/labelled/"
)
preprocess(
    {k:v for k,v in patient_info.items() if inSplit(k, splits["train"]["ulab"])},
    spacing_target, "preprocessed/training/unlabelled/"
)
preprocess(
    {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
    spacing_target, "preprocessed/validation/"
)
preprocess(
    {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
    spacing_target, "preprocessed/soft_validation/", soft_preprocessing=True
)
preprocess(
    {k:v for k,v in patient_info.items() if inSplit(k, splits["test"])},
    spacing_target, "preprocessed/testing/", soft_preprocessing=True
)


# ## $\mathcal{M}$ - Supervised Training

# In[ ]:


import torch.nn as nn
import os
import torch

from baseline_1 import Baseline_1
from unet_model import Baseline_2
from utils import AttrDict
from utils import GDLoss, CELoss
from utils import device
from utils import Validator, Checkpointer
from utils import supervised_training
from utils import ACDCDataLoader, ACDCAllPatients
from utils import BATCH_SIZE, EPOCHS, CKPT
from utils import transform_augmentation_downsample, transform
from utils import plot_history


# In[ ]:


Model = Baseline_2

model = nn.ModuleDict([
    [axis, Model(
        AttrDict(**{
            "lr": 0.01,
            "functions": [GDLoss, CELoss]
        })
    )] for axis in ["SA", "LA"]
]).to(device)

ckpts = None
if ckpts is not None:
    for axis, ckpt in ckpts.items():
        _, start = os.path.split(ckpt)
        start = int(start.replace(".pth", ""))
        ckpt = torch.load(ckpt)
        model[axis].load_state_dict(ckpt["M"])
        model[axis].optimizer.load_state_dict(ckpt["M_optim"])
else:
    start = 1
print(model)


# In[ ]:


validators = {
    "SA": Validator(5),
    "LA": Validator(5)
}

for axis in ["SA", "LA"]:
    supervised_training(
        model[axis],
        range(start, EPOCHS),
        torch.utils.data.DataLoader(
            ACDCAllPatients(
                os.path.join("preprocessed/training/labelled/", axis),
                transform=transform_augmentation_downsample
            ),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        ACDCDataLoader(
            os.path.join("preprocessed/validation/", axis),
            batch_size=BATCH_SIZE, transform=transform
        ),
        validators[axis],
        Checkpointer(os.path.join(CKPT, "M", axis))
    )

    plot_history(validators[axis].get_history("val"), "M-supervised-training")
    torch.save(model.state_dict(), './model_saved.pth')


# ## $\mathcal{M}$ - Testing

# In[ ]:



import os
import torch
import pickle
import numpy as np
import pandas as pd

from baseline_1 import Baseline_1
from unet_model import Baseline_2
from utils import device
from utils import ACDCDataLoader
from utils import BATCH_SIZE, CKPT
from utils import transform
from utils import infer_predictions
from utils import get_splits
from utils import postprocess_predictions, display_results


# In[ ]:


Model = Baseline_2

model = nn.ModuleDict([
    [axis, Model()] for axis in ["SA", "LA"]
]).to(device)

for axis in ["SA", "LA"]:
    ckpt = os.path.join(CKPT, "M", axis, "best_000.pth")
    model[axis].load_state_dict(torch.load(ckpt)["M"])
    model[axis].to(device)
    model.eval()

    infer_predictions(
        os.path.join("inference", axis),
        ACDCDataLoader(
            f"preprocessed/testing/{axis}",
            batch_size = BATCH_SIZE,
            transform = transform,
            transform_gt = False
        ),
        model[axis]
    )


# In[ ]:


splits = get_splits(os.path.join(CKPT, "splits.pkl"))

with open(os.path.join("preprocessed", "patient_info.pkl"),'rb') as f:
    patient_info = pickle.load(f)

spacings = [
    patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (
        splits["train"]["lab"] + splits["train"]["ulab"] + splits["val"]
    )
]

current_spacing = np.percentile(np.vstack(spacings), 50, 0)


# In[ ]:


results = {}
for axis in ["SA", "LA"]:
    results[axis] = postprocess_predictions(
        os.path.join("inference", axis),
        patient_info,
        current_spacing,
        os.path.join("postprocessed", axis),
    )

with open("postprocessed/results.pkl", "wb") as f:
    pickle.dump(results,f)

display_results(results)


# Here we test our model on the testing set provided in the M&Ms-2 Challenge. For this reason, no GT is available, and validation metrics cannot be directly evaluated (this is why all values in the table above are NaN). The code in the cells below display the results reported in the <a href="https://www.ub.edu/mnms-2/#:~:text=the%20competition%20in-,Codalab,-to%20submit%20your"> Codalab platform</a>.

# In[ ]:


for axis in ["SA", "LA"]:
    for src in os.listdir(os.path.join("postprocessed", axis)):
        id = src.split("_")[0]
        if int(id) < 161:
            continue
        nib_image = nib.load(os.path.join("postprocessed", axis, src))
        image = np.around(nib_image.get_fdata()).astype(int)
        image = np.where(image==3, 1, 0)
        dst = os.path.join("submission", id, src.split(".nii.gz")[0] + "_pred.nii.gz")
        if not os.path.isdir(os.path.split(dst)[0]):
            os.makedirs(os.path.split(dst)[0])
        nib.save(nib.Nifti1Image(image, nib_image.affine, nib_image.header), dst)


# In[ ]:


# In[ ]:


print("\033[1mBaseline\033[0m")
pd.DataFrame.from_dict({
    "axis": ["SA", "LA", "avg"],
    "RV_DC": [0.903681832739, 0.899725671050, 0.902692792316],
    "RV_HD": [13.350667610394, 7.404528745594, 11.864132894194],
}).set_index("axis")


# ## $\mathcal{R}$ - Training

# In[ ]:


import torch.nn as nn
import torch
import os

from reconstructor import Reconstructor
from utils import AttrDict
from utils import device
from utils import plot_history
from utils import ACDCAllPatients, ACDCDataLoader
from utils import transform
from utils import BATCH_SIZE, CKPT


# In[ ]:


ae = nn.ModuleDict([
    [axis, Reconstructor(
        AttrDict(**{
            "latent_size": 100,
            "lr": 2e-4,
            "last_layer": [4,2,1],
            "in_channels": 4,
            "weighted_epochs": 0
        })
    )] for axis in ["SA", "LA"]
]).to(device)

ckpts = None
if ckpts is not None:
    for axis, ckpt in ckpts.items():
        _, start = os.path.split(ckpt)
        start = int(start.replace(".pth", ""))
        ckpt = torch.load(ckpt)
        ae[axis].load_state_dict(ckpt["R"])
        ae[axis].optimizer.load_state_dict(ckpt["R_optim"])
else:
    start = 0
print(ae)


# In[ ]:


for axis in ["SA", "LA"]:
    plot_history(ae[axis].training_routine(
        range(start, 500),
        torch.utils.data.DataLoader(
            ACDCAllPatients(
                os.path.join("preprocessed/training/labelled/", axis),
                transform=transform
            ),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        ACDCDataLoader(
            os.path.join("preprocessed/validation/", axis),
            batch_size=BATCH_SIZE, transform=transform
        ),
        os.path.join(CKPT, "R", axis)
    ), "R-training")


# ## QC-based Candidate Selection

# In[ ]:


import torch.nn as nn
import os
import torch
import pickle
import numpy as np

from baseline_1 import Baseline_1
from unet_model import Baseline_2
from reconstructor import Reconstructor
from utils import device
from utils import AttrDict
from utils import Validator
from utils import ACDCDataLoader
from utils import BATCH_SIZE, CKPT
from utils import transform
from utils import GDLoss, CELoss, GDLoss_RV, CELoss_RV
from utils import infer_predictions
from utils import get_splits
from utils import postprocess_predictions
from utils import display_results


# In[ ]:


Model = Baseline_2

model = nn.ModuleDict([
    [axis, Model(
        AttrDict(**{
            "lr": 0.01,

            "functions": [GDLoss, CELoss],
            "functions_RV": [GDLoss_RV, CELoss_RV]
        })
    )] for axis in ["SA", "LA"]
]).to(device)

ae = nn.ModuleDict([
    [axis, Reconstructor(
        AttrDict(**{
            "latent_size": 100,
            "lr": 2e-4,
            "last_layer": [4,2,1],
            "in_channels": 4,
            "weighted_epochs": 0
        })
    )] for axis in ["SA", "LA"]
]).to(device)

validators = {
    "SA": Validator(5),
    "LA": Validator(5)
}

for axis in ["SA", "LA"]:
    ckpt = os.path.join(CKPT, "R", axis)
    ckpt = os.path.join(ckpt, sorted([file for file in os.listdir(ckpt) if "_best" in file])[-1])
    ckpt = torch.load(ckpt)
    ae[axis].load_state_dict(ckpt["R"])
    ae.eval()
    
    ckpt = os.path.join(CKPT, "M_refinement")
    if not os.path.isdir(ckpt):
        ckpt = os.path.join(CKPT, "M")
    for file in os.listdir(os.path.join(ckpt, axis)):
        if "best_" not in file or not file.endswith(".pth"):
            continue
        model[axis].load_state_dict(torch.load(os.path.join(ckpt, axis, file))["M"])
        model.eval()
        with torch.no_grad():
            validators[axis].domain_evaluation(
                "test",
                model[axis],
                ACDCDataLoader(
                    f"preprocessed/testing/{axis}",
                    batch_size=BATCH_SIZE,
                    transform=transform,
                    transform_gt=False
                ),    
                reconstructor=ae[axis]
            )


# In[ ]:


for axis in ["SA", "LA"]:
    infer_predictions(
        os.path.join("inference", axis),
        ACDCDataLoader(
            f"preprocessed/testing/{axis}",
            batch_size=BATCH_SIZE,
            transform=transform,
            transform_gt=False
        ),
        validator=validators[axis]
    )


# In[ ]:


splits = get_splits(os.path.join(CKPT, "splits.pkl"))

with open(os.path.join("preprocessed", "patient_info.pkl"),'rb') as f:
    patient_info = pickle.load(f)

spacings = [
    patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (
        splits["train"]["lab"] + splits["train"]["ulab"] + splits["val"]
    )
]

current_spacing = np.percentile(np.vstack(spacings), 50, 0)


# In[ ]:


results = {}
for axis in ["SA", "LA"]:
    results[axis] = postprocess_predictions(
        os.path.join("inference", axis),
        patient_info,
        current_spacing,
        os.path.join("postprocessed", axis),
    )

with open("postprocessed/results.pkl", "wb") as f:
    pickle.dump(results,f)

display_results(results)


# In[ ]:


for axis in ["SA", "LA"]:
    for src in os.listdir(os.path.join("postprocessed", axis)):
        id = src.split("_")[0]
        if int(id) < 161:
            continue
        nib_image = nib.load(os.path.join("postprocessed", axis, src))
        image = np.around(nib_image.get_fdata()).astype(int)
        image = np.where(image==3, 1, 0)
        dst = os.path.join("submission", id, src.split(".nii.gz")[0] + "_pred.nii.gz")
        if not os.path.isdir(os.path.split(dst)[0]):
            os.makedirs(os.path.split(dst)[0])
        nib.save(nib.Nifti1Image(image, nib_image.affine, nib_image.header), dst)


# In[ ]:


# We display below the results reported in the <a href="https://www.ub.edu/mnms-2/#:~:text=the%20competition%20in-,Codalab,-to%20submit%20your"> Codalab platform</a> after submitting the .zip file generated above.

# In[ ]:import torch.nn as nn


print("\033[1mQC-based Candidate Selection\033[0m")
pd.DataFrame.from_dict({
    "axis": ["SA", "LA", "avg"],
    "RV_DC": [0.898024197115, 0.904328292327, 0.899600220918],
    "RV_HD": [12.194214048970, 7.522092252961, 11.026183599968],
}).set_index("axis")


# ## Semi-Supervised Refinement

# In[ ]:


import torch.nn as nn
import os
import torch
import pickle

from baseline_1 import Baseline_1
from unet_model import Baseline_2
from reconstructor import Reconstructor
from utils import AttrDict
from utils import GDLoss, CELoss, GDLoss_RV, CELoss_RV
from utils import device
from utils import Validator, Checkpointer
from utils import semisupervised_refinement
from utils import ACDCDataLoader, ACDCAllPatients
from utils import BATCH_SIZE, EPOCHS, CKPT
from utils import transform_augmentation_downsample, transform
from utils import plot_history


# In[ ]:


Model = Baseline_2

model = nn.ModuleDict([
    [axis, Model(
        AttrDict(**{
            "lr": 0.01,
            "functions": [GDLoss, CELoss],
            "functions_RV": [GDLoss_RV, CELoss_RV]
        })
    )] for axis in ["SA", "LA"]
]).to(device)

ae = nn.ModuleDict([
    [axis, Reconstructor(
        AttrDict(**{
            "latent_size": 100,
            "lr": 2e-4,
            "last_layer": [4,2,1],
            "in_channels": 4,
            "weighted_epochs": 0
        })
    )] for axis in ["SA", "LA"]
]).to(device)

validators = {
    "SA": Validator(5),
    "LA": Validator(5)
}

for axis in ["SA", "LA"]:
    ckpt = os.path.join(CKPT, "R", axis)
    ckpt = os.path.join(ckpt, sorted([file for file in os.listdir(ckpt) if "_best" in file])[-1])
    ckpt = torch.load(ckpt)
    ae[axis].load_state_dict(ckpt["R"])
    ae.eval()

    ckpt = os.path.join(CKPT, "M", axis, "200.pth")
    _, start = os.path.split(ckpt)
    start = int(start.replace(".pth", ""))
    ckpt = torch.load(ckpt)
    model[axis].load_state_dict(ckpt["M"])
    model[axis].optimizer.load_state_dict(ckpt["M_optim"])

    ckpt = os.path.join(CKPT, "M", axis, "200_val.pkl")
    with open(ckpt, "rb") as f:
        validators[axis] = pickle.load(f)

print(model)


# In[ ]:


for axis in ["SA", "LA"]:
    semisupervised_refinement(
        model[axis],
        ae[axis],
        range(start, EPOCHS),
        torch.utils.data.DataLoader(
            ACDCAllPatients(
                os.path.join("preprocessed/training/labelled/", axis),
                transform=transform_augmentation_downsample
            ),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        ACDCDataLoader(
            os.path.join("preprocessed/validation/", axis),
            batch_size=BATCH_SIZE, transform=transform
        ),
        ACDCDataLoader(
            os.path.join("preprocessed/training/unlabelled/", axis),
            batch_size = BATCH_SIZE, transform=transform
        ),
        validators[axis],
        Checkpointer(os.path.join(CKPT, "M_refinement", axis))
    )

    plot_history(validators[axis].get_history("val"), "Semi-Supervised-Refinement")


# After semi-supervised refinement, go back to QC-based Candidate Selection to validate your model. We report below the results from the <a href="https://www.ub.edu/mnms-2/#:~:text=the%20competition%20in-,Codalab,-to%20submit%20your"> Codalab platform</a>.

# In[ ]:


print("\033[1mSemi-Supervised Refinement\033[0m")
pd.DataFrame.from_dict({
    "axis": ["SA", "LA", "avg"],
    "RV_DC": [0.900794533587, 0.899223080762, 0.900401670381],
    "RV_HD": [12.268410479696, 7.016155479374, 10.955346729616],
}).set_index("axis")

