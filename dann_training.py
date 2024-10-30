# import random
# import numpy as np
# import torch

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# import os
# from dann_utils import get_vendor_info
# from dann_utils import get_splits
# from utils import generate_patient_info, crop_image
# from utils import preprocess, preprocess_image, inSplit

# vendor_info = get_vendor_info("./data/MnM2/dataset_information.csv")
# vendor_info = vendor_info.sample(frac=1).reset_index(drop=True)

# if not os.path.isdir("dann_preprocessed"):
#     os.makedirs("dann_preprocessed")

# splits = get_splits(vendor_info, os.path.join("dann_preprocessed", "splits.pkl"))

# patient_info = generate_patient_info(vendor_info, os.path.join("dann_preprocessed", "patient_info.pkl"))

# spacings = [
#     patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (
#         splits["train"] + splits["train"] + splits["val"]
#     )
# ]
# spacing_target = np.percentile(np.vstack(spacings), 50, 0)

# if not os.path.isdir("dann_preprocessed/training/"): os.makedirs("dann_preprocessed/training/")
# if not os.path.isdir("dann_preprocessed/validation/"): os.makedirs("dann_preprocessed/validation/")
# if not os.path.isdir("dann_preprocessed/soft_validation/"): os.makedirs("dann_preprocessed/soft_validation/")
# if not os.path.isdir("dann_preprocessed/testing/"): os.makedirs("dann_preprocessed/testing/")

# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["train"])},
#     spacing_target, "dann_preprocessed/training/"
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
#     spacing_target, "dann_preprocessed/validation/"
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
#     spacing_target, "dann_preprocessed/soft_validation/", soft_preprocessing=True
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["test"])},
#     spacing_target, "dann_preprocessed/testing/", soft_preprocessing=True
# )

# import torch.nn as nn
# import os
# import torch

# from baseline_1 import Baseline_1
# from unet_model import Baseline_DANN
# from utils import AttrDict
# from utils import GDLoss, CELoss
# from utils import device
# from utils import Validator, Checkpointer
# from utils import dann_training
# from dann_loader import DANNDataLoader, DANNAllPatients
# from utils import BATCH_SIZE, EPOCHS, CKPT
# from utils import transform_augmentation_downsample, transform
# from utils import plot_history


# Model = Baseline_DANN

# model = nn.ModuleDict([
#     [axis, Model(
#         AttrDict(**{
#             "lr": 0.01,
#             "functions": [GDLoss, CELoss]
#         })
#     )] for axis in ["SA", "LA"]
# ]).to(device)

# ckpts = None
# if ckpts is not None:
#     for axis, ckpt in ckpts.items():
#         _, start = os.path.split(ckpt)
#         start = int(start.replace(".pth", ""))
#         ckpt = torch.load(ckpt)
#         model[axis].load_state_dict(ckpt["M_dann"])
#         model[axis].optimizer.load_state_dict(ckpt["M_dann_optim"])
# else:
#     start = 1

# validators = {
#     "SA": Validator(5),
#     "LA": Validator(5)
# }

# for axis in ["SA", "LA"]:
#     dann_training(
#         model[axis],
#         range(start, EPOCHS),
#         torch.utils.data.DataLoader(
#             DANNAllPatients(
#                 os.path.join("dann_preprocessed/training/", axis),
#                 transform=transform_augmentation_downsample
#             ),
#             batch_size=BATCH_SIZE, shuffle=False, num_workers=0
#         ),
#         DANNDataLoader(
#             os.path.join("dann_preprocessed/validation/", axis),
#             batch_size=BATCH_SIZE, transform=transform
#         ),
#         validators[axis],
#         Checkpointer(os.path.join(CKPT, "M_dann", axis))
#     )

#     plot_history(validators[axis].get_history("val"), "M-dann " + axis)

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
from utils import generate_patient_info, crop_image
from utils import preprocess, preprocess_image, inSplit
from utils import get_splits_supervised


# vendor_info = get_vendor_info("./data/MnM2/dataset_information.csv")
# vendor_info 


# # In[5]:


# vendor_info[['VENDOR','SUBJECT_CODE']].groupby(['VENDOR']).count().rename(columns={'SUBJECT_CODE':'NUM_STUDIES'})


# # In[6]:


# if not os.path.isdir("preprocessed"):
#     os.makedirs("preprocessed")
# splits = get_splits_supervised(vendor_info, os.path.join("dann_preprocessed", "splits.pkl"))


# # In[7]:


# patient_info = generate_patient_info(vendor_info, os.path.join("preprocessed", "patient_info.pkl"))


# # In[ ]:


# spacings = [
#     patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (splits["train"] + splits["val"])
# ]
# spacing_target = np.percentile(np.vstack(spacings), 50, 0)

# if not os.path.isdir("preprocessed/training"): os.makedirs("preprocessed/training")
# if not os.path.isdir("preprocessed/validation/"): os.makedirs("preprocessed/validation/")
# if not os.path.isdir("preprocessed/soft_validation/"): os.makedirs("preprocessed/soft_validation/")
# if not os.path.isdir("preprocessed/testing/"): os.makedirs("preprocessed/testing/")

# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["train"])},
#     spacing_target, "preprocessed/training/"
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
#     spacing_target, "preprocessed/validation/"
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["val"])},
#     spacing_target, "preprocessed/soft_validation/", soft_preprocessing=True
# )
# preprocess(
#     {k:v for k,v in patient_info.items() if inSplit(k, splits["test"])},
#     spacing_target, "preprocessed/testing/", soft_preprocessing=True
# )


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
                os.path.join("preprocessed/training/", axis),
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

    plot_history(validators[axis].get_history("val"), f"M-supervised-training-{axis}")


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


splits = get_splits_supervised(vendor_info, os.path.join("dann_preprocessed", "splits.pkl"))

with open(os.path.join("preprocessed", "patient_info.pkl"),'rb') as f:
    patient_info = pickle.load(f)

spacings = [
    patient_info["{:03d}_{}".format(id, axis)]["spacing"] for axis in ["SA", "LA"] for id in (
        splits["train"] + splits["val"]
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