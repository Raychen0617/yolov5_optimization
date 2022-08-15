
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model, Detect
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLossQuant, ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

from optimizer.convert_compare import convert_and_compare
from torchvision import datasets, transforms
from utils.dataloaders import create_cifar
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.util.train_util import DLContext, get_device, train



device = "cpu"
dummy_input = torch.rand(1, 3, 640, 640)
model = torch.load('./checkpoint/yolov5s_nas.pt')['model'].float()

for k, m in model.named_modules():
    if isinstance(m, Detect):
        print(m.inplace, m.onnx_dynamic, m.export)
        m.inplace = False
        m.onnx_dynamic = True
        m.export = True

convert_and_compare(model, './checkpoint/yolov5s_inplace_false.tflite', dummy_input)

'''
quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'asymmetric': True, 'per_tensor': True})
qat_model = quantizer.quantize()
qat_model.to(device=device)

with torch.no_grad():  
    qat_model.cpu()
    qat_model = torch.quantization.convert(qat_model)
    torch.backends.quantized.engine = quantizer.backend
    convert_and_compare(qat_model, './checkpoint/best.tflite', dummy_input)
'''