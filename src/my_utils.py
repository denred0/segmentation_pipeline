import random
import os
import numpy as np
import torch


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
