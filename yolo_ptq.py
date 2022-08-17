
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
from utils.dataloaders import create_dataloader, create_cifar
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
from tinynn.graph.quantization.quantizer import QATQuantizer, PostQuantizer
from tinynn.util.train_util import DLContext, get_device, train, AverageMeter


def calibrate(model, context: DLContext):

    model.to(device=context.device)
    model.eval()

    avg_batch_time = AverageMeter()

    with torch.no_grad():
        end = time.time()
        #print("********", len(context.train_loader))
        for i, (image, targets, paths, _) in enumerate(context.train_loader):

            if context.max_iteration is not None and i >= context.max_iteration:
                break

            image = image.to(device=context.device, non_blocking=True).float() / 255

            model(image)

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(f'Calibrate: [{i}/{len(context.val_loader)}]\tTime {avg_batch_time.avg:.5f}\t')

            context.iteration += 1
            

device = "cuda"
dummy_input = torch.rand(1, 3, 640, 640)
model = torch.load('./checkpoint/yolov5s_nas.pt')['model'].float()

for k, m in model.named_modules():
    if isinstance(m, Detect):
        print(m.inplace, m.onnx_dynamic, m.export)
        m.inplace = False
        m.onnx_dynamic = True
        m.export = True

#convert_and_compare(model, './checkpoint/yolov5s_inplace_false.tflite', dummy_input)
quantizer = PostQuantizer(model, dummy_input, work_dir='./quantization')
ptq_model = quantizer.quantize()
context = DLContext()
context.device = device
context.max_iteration = 100


# Image size
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(640, gs, floor=gs * 2)  # verify imgsz is gs-multiple



with open('./data/hyps/hyp.scratch-low.yaml', errors='ignore') as f:
    hyp = yaml.safe_load(f)  # load hyps dict
    if 'anchors' not in hyp:  # anchors commented in hyp.yaml
        hyp['anchors'] = 3

nc = 80
nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
hyp['label_smoothing'] = 0.0

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

# Trainloader
context.train_loader, _ = create_dataloader("/root/datasets/coco/images/train2017",
                                          imgsz,
                                          64 // WORLD_SIZE,
                                          gs,
                                          single_cls=False,
                                          hyp=hyp,
                                          augment=True,
                                          cache=None,
                                          rect=False,
                                          rank=LOCAL_RANK,
                                          workers=0,
                                          image_weights=False,
                                          quad=False,
                                          prefix=colorstr('train: '),
                                          shuffle=True)

context.val_loader  = create_dataloader("/root/datasets/coco/images/val2017",
                                       imgsz,
                                       64 // WORLD_SIZE * 2,
                                       gs,
                                       single_cls=False,
                                       hyp=hyp,
                                       cache=None,
                                       rect=True,
                                       rank=-1,
                                       workers=0,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

# Post quantization calibration
calibrate(ptq_model, context)

with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model = torch.quantization.convert(ptq_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        print("Converting ............")
        convert_and_compare(ptq_model, './checkpoint/yolov5s_ptq.tflite', dummy_input)
        #converter = TFLiteConverter(ptq_model, dummy_input, tflite_path='out/qat_model.tflite')
        #converter.convert()

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