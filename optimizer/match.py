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
from models.yolo import Backbone, Model, NASBACKBONE
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript
from torchvision import transforms
from models.common import NASConv, NASC3

'''
Main: 
Match two pytorch models with same structure, but different hyperparameters (ex: input channels, output channels)

'''

def extract_backbone():
    
    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)

    if isnas:

        model = NASBACKBONE(cfg=nas_model, nc=200).to(device=device).backbone.model
        nasjson = json.load(open(nas_json))
        # fixed nas architecture 
        for i in range(len(nas)):
            if isinstance(nas[i], NASConv):
                nas[i].conv = nas[i].conv[nasjson["nasconv_{}".format(i)]]
            elif isinstance(nas[i], NASC3) :
                nas[i].cv1.conv = nas[i].cv1.conv[nasjson["nasconv_{}-1".format(i)]]
                nas[i].cv2.conv = nas[i].cv2.conv[nasjson["nasconv_{}-2".format(i)]]

    else:
        model = torch.load(model_backbone_checkpoint).to(device=device).backbone


def fixed_nas(nas_backbone, nas_json):

    # This function fixed nas backbone with given json file 

    nasjson = json.load(open(nas_json))

    # fixed nas architecture 
    for i in range(len(nas_backbone)):
        if isinstance(nas_backbone[i], NASConv):
            nas_backbone[i].conv = nas_backbone[i].conv[nasjson["nasconv_{}".format(i)]]
        elif isinstance(nas_backbone[i], NASC3) :
            nas_backbone[i].cv1.conv = nas_backbone[i].cv1.conv[nasjson["nasconv_{}-1".format(i)]]
            nas_backbone[i].cv2.conv = nas_backbone[i].cv2.conv[nasjson["nasconv_{}-2".format(i)]]
    
    return nas_backbone


# This function construct yolov5 - nas by providing  .json (choice) and .yaml 
def match_nas(yolo, nas_backbone, nas_json, save, fixed=True):

    device = torch.device("cpu")
    
    # if need to fix nas backbone 
    if fixed:
        nas = fixed_nas(nas_backbone, nas_json)
    else:
        nas = nas_backbone
    
    dummy_input = torch.rand(1, 3, 640, 640)

    # store nas layers
    nas_layer = {}
    for name, model_type in nas.named_modules():
        if isinstance(model_type, Conv) or isinstance(model_type, NASConv) or isinstance(model_type, NASC3):
            nas_layer[name] = model_type

    for name, model_type in yolo.model.named_modules():
        
        if isinstance(model_type, Conv) or isinstance(model_type, NASConv) or isinstance(model_type, NASC3):
            if nas_layer.get(name) is None:
                    print(name, "cannot be found")
                    # layer 9 ~ 24 not found is normal 

            elif nas_layer[name].conv.kernel_size != model_type.conv.kernel_size  or \
            nas_layer[name].conv.stride != model_type.conv.stride or  nas_layer[name].conv.padding != model_type.conv.padding \
            or nas_layer[name].conv.in_channels != model_type.conv.in_channels or nas_layer[name].conv.out_channels != model_type.conv.out_channels:
                print("*****", name, nas_layer[name])
                model_type.conv = nas_layer[name].conv
                model_type.bn = nas_layer[name].bn

    # retune model size for cspnet concat 
    yolo.model[9].cv1.conv = nn.Conv2d(yolo.model[8].cv3.conv.out_channels, yolo.model[9].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[10].conv = nn.Conv2d(yolo.model[9].cv2.conv.out_channels, yolo.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv1.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv2.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv1.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv2.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.to(device=device)

    # load nas state dict back 
    yolo.model.load_state_dict(nas.state_dict(), strict=False)
    print("Matching two different models ", yolo(dummy_input)[0].shape == (1, 3, 80, 80, 85))
    torch.save(yolo, save)
    print("Save at ", save)
    return yolo



def match(yolo, pruned_yolo, save):
    
    # This function matches pruned yolo with original yolo 
    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)
    #pruned_yolo = torch.load(pruned_model).to(device=device)
    #pruned_yolo = pruned_yolo.backbone

    pruned_yolo_layer = {}
    for name, model_type in pruned_yolo.named_modules():
        if isinstance(model_type, Conv):
            pruned_yolo_layer[name] = model_type


    for name, model_type in yolo.named_modules():
        if isinstance(model_type, Conv):

            if pruned_yolo_layer.get(name) is None:
                print(name, "cannot be found")

            elif pruned_yolo_layer[name].conv.in_channels != model_type.conv.in_channels  or pruned_yolo_layer[name].conv.out_channels != model_type.conv.out_channels:
                model_type.conv = pruned_yolo_layer[name].conv
                model_type.bn = pruned_yolo_layer[name].bn


    # retune model size for cspnet concat 
    yolo.model[9].cv1.conv = nn.Conv2d(yolo.model[8].cv3.conv.out_channels, yolo.model[9].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[10].conv = nn.Conv2d(yolo.model[9].cv2.conv.out_channels, yolo.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv1.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv2.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv1.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv2.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.to(device=device)

    # load pruned state dict back 
    yolo.load_state_dict(pruned_yolo.state_dict() , strict=False)

    print("Matching two different models ", yolo(dummy_input)[0].shape == (1, 3, 80, 80, 85))

    torch.save(yolo, save)
    print("Save at ", save)
    return yolo

if __name__ == '__main__':
    match_nas()


