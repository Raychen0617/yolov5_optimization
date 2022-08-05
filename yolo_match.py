from pyexpat import model
from turtle import forward
from numpy import isin
import torch, torchvision
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner, L2NormPruner,FPGMPruner,ActivationAPoZRankPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from utils.general import check_img_size
from models.yolo import Detect
from utils.activations import SiLU
import torch.nn as nn
from nni.compression.pytorch.utils.counter import count_flops_params
import time 
from nni.compression.pytorch.utils import not_safe_to_prune
from models.yolo import Backbone, Model
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript
from torchvision import transforms

'''
Main: 
Match two pytorch models with same structure, but different hyperparameters (ex: input channels, output channels)

'''


def match():

    device = torch.device("cpu")
    yolo = Model('./models/yolov5s.yaml').to(device=device)
    dummy_input = torch.rand(1, 3, 640, 640)
    pruned_yolo = torch.load("./checkpoint/pruned_yolov5s.pt").to(device=device)


    pruned_yolo_layer = {}
    for name, model_type in pruned_yolo.named_modules():
        if isinstance(model_type, Conv):
            pruned_yolo_layer[name] = model_type


    for name, model_type in yolo.named_modules():
        if isinstance(model_type, Conv):
            if pruned_yolo_layer[name].conv.in_channels != model_type.conv.in_channels  or pruned_yolo_layer[name].conv.out_channels != model_type.conv.out_channels:
                model_type.conv = pruned_yolo_layer[name].conv
                model_type.bn = pruned_yolo_layer[name].bn


    # retune model size for cspnet concat 
    yolo.model[10].conv = nn.Conv2d(yolo.model[9].cv2.conv.out_channels, yolo.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv1.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv2.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv1.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv2.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.to(device=device)

    print("Matching two different models ", yolo(dummy_input)[0].shape == (1, 3, 80, 80, 85))


if __name__ == '__main__':
    match()


