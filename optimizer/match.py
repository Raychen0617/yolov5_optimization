from operator import mod
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

# the function is been implemented by nni (fixed_arch), it will be deprecated in the future
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
def match_nas(yolo, nas_backbone, save):

    device = torch.device("cpu") 
    dummy_input = torch.rand(1, 3, 640, 640)

    # store nas layers
    nas_backbone_layer = []
    for module in nas_backbone.backbone.model.children():
        if isinstance(module, NASConv) or isinstance(module, NASC3):
            nas_backbone_layer.append(module)

    # match nas layer with yolov5's backbone
    for i in range(9):
        if isinstance(yolo.model[i], Conv) and isinstance(nas_backbone_layer[i], NASConv):
            yolo.model[i].conv = nas_backbone_layer[i].conv
            yolo.model[i].bn = nas_backbone_layer[i].bn
            yolo.model[i].act = nas_backbone_layer[i].act
    
        elif isinstance(yolo.model[i], C3) and isinstance(nas_backbone_layer[i], NASC3):
            yolo.model[i].cv1.conv = nas_backbone_layer[i].total.cv1.conv
            yolo.model[i].cv1.bn = nas_backbone_layer[i].total.cv1.bn
            yolo.model[i].cv1.act = nas_backbone_layer[i].total.cv1.act
            yolo.model[i].cv2.conv = nas_backbone_layer[i].total.cv2.conv
            yolo.model[i].cv2.bn = nas_backbone_layer[i].total.cv2.bn
            yolo.model[i].cv2.act = nas_backbone_layer[i].total.cv2.act
            yolo.model[i].cv3.conv = nas_backbone_layer[i].total.cv3.conv
            yolo.model[i].cv3.bn = nas_backbone_layer[i].total.cv3.bn
            yolo.model[i].cv3.act = nas_backbone_layer[i].total.cv3.act
            yolo.model[i].m = nas_backbone_layer[i].total.m
        else:
            print("!!!!!!!!!!!!     error causing when matching     !!!!!!!!!!!!!!")

    # retune model size for backbone concat neck  
    yolo.model[9].cv1.conv = nn.Conv2d(yolo.model[8].cv3.conv.out_channels, yolo.model[9].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[10].conv = nn.Conv2d(yolo.model[9].cv2.conv.out_channels, yolo.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv1.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv2.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv1.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv2.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.to(device=device)

    # load nas state dict back 
    #yolo.model.load_state_dict(nas_backbone.state_dict(), strict=False)
    
    print("Matching two different models ", yolo(dummy_input)[0].shape == (1, 3, 80, 80, 85))
    torch.save(yolo, save)
    print("Save at ", save)
    
    return yolo

def conv_match(new, old):
    new.conv = old.conv
    new.act = old.act
    new.bn = old.bn
    return new

def fix_nasbackbone(backbone, backbone_nas):
    
    # This function matches nas backbone to original backbone

    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)

    for name, module in backbone_nas.named_children():
        backbone_nas_layer = []
        for submodule in module:
            backbone_nas_layer.append(submodule)

    for name, module in backbone.named_children():
        
        for layerid in range(len(backbone_nas_layer)):
            #print(module[layerid], type(module[layerid]))
            if isinstance(module[layerid], Conv):
                module[layerid] = conv_match(module[layerid], backbone_nas_layer[layerid])
            
            elif isinstance(module[layerid], C3):
                module[layerid].cv1 = conv_match(module[layerid].cv1, backbone_nas_layer[layerid].total.cv1)
                module[layerid].cv2 = conv_match(module[layerid].cv2, backbone_nas_layer[layerid].total.cv2)
                module[layerid].cv3 = conv_match(module[layerid].cv3, backbone_nas_layer[layerid].total.cv3)
                module[layerid].m = backbone_nas_layer[layerid].total.m

            else:
                print("sth gets wrong !!!!, please checkout optimizer/match.py")
                exit()
    return backbone

def fix_nasyolo(yolo, nas_yolo, backbone_layer=9):
    
    # This function matches nas yolo to original yolo 

    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)

    for name, module in nas_yolo.named_children():
        nas_yolo_layer = []
        for submodule in module:
            nas_yolo_layer.append(submodule)

    for name, module in yolo.named_children():
        
        for layerid in range(backbone_layer):
            
            if isinstance(module[layerid], Conv):
                module[layerid] = conv_match(module[layerid], nas_yolo_layer[layerid])
            
            elif isinstance(module[layerid], C3):
                module[layerid].cv1 = conv_match(module[layerid].cv1, nas_yolo_layer[layerid].total.cv1)
                module[layerid].cv2 = conv_match(module[layerid].cv2, nas_yolo_layer[layerid].total.cv2)
                module[layerid].cv3 = conv_match(module[layerid].cv3, nas_yolo_layer[layerid].total.cv3)
                module[layerid].m = nas_yolo_layer[layerid].total.m

            else:
                print("sth gets wrong !!!!, please checkout optimizer/match.py")
                exit()
    return yolo


def extract_backbone(backbone, yolo, backbone_layer=9):
    
    # This function extracts Yolo's backbone 

    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)

    for name, module in yolo.named_children():
        yolo = []
        for submodule in module:
            yolo.append(submodule)

    for name, module in backbone.named_children():
        
        for layerid in range(backbone_layer):
            #print(module[layerid], type(module[layerid]))
            if isinstance(module[layerid], Conv):
                module[layerid] = conv_match(module[layerid], yolo[layerid])
            
            elif isinstance(module[layerid], C3):
                module[layerid].cv1 = conv_match(module[layerid].cv1, yolo[layerid].cv1)
                module[layerid].cv2 = conv_match(module[layerid].cv2, yolo[layerid].cv2)
                module[layerid].cv3 = conv_match(module[layerid].cv3, yolo[layerid].cv3)
                module[layerid].m = yolo[layerid].m

            else:
                print("sth gets wrong !!!!, please checkout optimizer/match.py")
                exit()
    return backbone


def match(yolo, pruned_yolo, save):
    
    # This function matches pruned yolo with original yolo 
    device = torch.device("cpu")
    dummy_input = torch.rand(1, 3, 640, 640)
    #pruned_yolo = torch.load(pruned_model).to(device=device)
    #pruned_yolo = pruned_yolo.backbone
    
    pruned_yolo_layer = {}
    for name, model_type in pruned_yolo.named_modules():
        if isinstance(model_type, NASConv) or isinstance(model_type, NASC3) or isinstance(model_type, Conv):
            pruned_yolo_layer[name] = model_type

    for name, model_type in yolo.named_modules():
        if isinstance(model_type, Conv):
            
            if pruned_yolo_layer.get(name) is None:
                print(name, "cannot be found")

            else:
                #print(name, pruned_yolo_layer[name], model_type)
                if pruned_yolo_layer[name].conv.in_channels != model_type.conv.in_channels  or pruned_yolo_layer[name].conv.out_channels != model_type.conv.out_channels \
                    or pruned_yolo_layer[name].conv.kernel_size != model_type.conv.kernel_size  or pruned_yolo_layer[name].conv.padding != model_type.conv.padding:
                    model_type.conv = pruned_yolo_layer[name].conv
                    
                if pruned_yolo_layer[name].bn != model_type.bn:
                    model_type.bn = pruned_yolo_layer[name].bn
                
                if pruned_yolo_layer[name].act != model_type.act:
                    model_type.act = pruned_yolo_layer[name].act
            
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
    
    if save is not None:
        torch.save(yolo, save)
        print("Save at ", save)
        
    return yolo

if __name__ == '__main__':
    match_nas()


