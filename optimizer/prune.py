from imageio import save
from numpy import isin
import torch, torchvision
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner, L2NormPruner,FPGMPruner,ActivationAPoZRankPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from utils.general import check_img_size
from models.yolo import Detect, BACKBONE
from utils.activations import SiLU
import torch.nn as nn
from nni.compression.pytorch.utils.counter import count_flops_params
import time 
from nni.compression.pytorch.utils import not_safe_to_prune
from models.yolo import Model
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript
from utils.dataloaders import create_tinyimagenet, create_cifar
import tqdm
from vision_toolbox import backbones

from utils.torch_utils import sparsity


def prune(model, save='./checkpoint/...', sparsity=0.25, method="L1"):
    
    device = torch.device("cuda:0")

    for k, m in model.named_modules(): 
        if isinstance(m, Conv): # assign export-friendly activations
            if isinstance(m, Detect):
                m.inplace = False
                m.onnx_dynamic = False

    imgsz = (64, 64)
    imgsz *= 2 if len(imgsz) == 1 else 1 # expand

    gs = 32 # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
    im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection


    cfg_list = [{
        'sparsity_per_layer': sparsity,
        'op_types': ['Conv2d'],
    },
    ]

    if method == "L1":
        pruner = L1NormPruner(model, cfg_list)
    elif method == "L2":
        pruner = L2NormPruner(model, cfg_list)
    elif method == "FPGM":
        pruner = FPGMPruner(model, cfg_list)
    else:
        print("Method is not supported !!! (prune.py)")
        return 
        
    _, masks = pruner.compress()
    pruner.show_pruned_weights()
    pruner._unwrap_model()

    m_speedup = ModelSpeedup(model, im, masks_file=masks)
    m_speedup.speedup_model()

    if save:
        print("Save at ", save)
        torch.save(model,save)
    return model

'''
def prune(ori_model='./models/yolov5sb.yaml', pretrain_backbone=backbones.darknet_yolov5s(pretrained=True), save='./checkpoint/...', sparsity=0.25, method="L1"):

    device = torch.device("cuda:0")
    model = BACKBONE(cfg=ori_model, nc=200).to(device=device)
    model.backbone.load_state_dict(convert_weights_direct(pretrain_backbone))

    for k, m in model.named_modules(): 
        if isinstance(m, Conv): # assign export-friendly activations
            if isinstance(m, Detect):
                m.inplace = False
                m.onnx_dynamic = False

    imgsz = (64, 64)
    imgsz *= 2 if len(imgsz) == 1 else 1 # expand

    gs = 32 # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
    im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection


    cfg_list = [{
        'sparsity_per_layer': sparsity,
        'op_types': ['Conv2d'],
    },
    ]

    if method == "L1":
        pruner = L1NormPruner(model, cfg_list)
    elif method == "L2":
        pruner = L2NormPruner(model, cfg_list)
    elif method == "FPGM":
        pruner = FPGMPruner(model, cfg_list)
    else:
        print("Method is not supported !!! (prune.py)")
        return 
        
    _, masks = pruner.compress()
    pruner.show_pruned_weights()
    pruner._unwrap_model()

    m_speedup = ModelSpeedup(model, im, masks_file=masks)
    m_speedup.speedup_model()

    print("Save at ", save)
    torch.save(model,save)
'''

if __name__ == '__main__':
    prune()


'''


device = torch.device("cuda:0")

# LOAD MODELS
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#model = attempt_load('yolov5s.pt', inplace=True, fuse=False, device=device) # load FP32 model
model = BACKBONE(cfg='./models/yolov5sb.yaml',nc=200).to(device=device)
model.backbone.load_state_dict(convert_weights_direct(backbones.darknet_yolov5s(pretrained=True)))
model.eval()


#exclude_layer = []
#after_spp = False
for k, m in model.named_modules(): 
    if isinstance(m, Conv): # assign export-friendly activations
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False

imgsz = (64, 64)
imgsz *= 2 if len(imgsz) == 1 else 1 # expand

gs = 32 # grid size (max stride)
imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection


cfg_list = [{
    'sparsity_per_layer': 0.25,
    'op_types': ['Conv2d'],
},
]

{
    'exclude': True,
    'op_names': exclude_layer
}


# test inference time before pruning
for _ in range(100):
    use_mask_out = model(im)
    
start = time.time()
for _ in range(100):
    use_mask_out = model(im)

origin_time = (time.time() - start)*100


pruner = L1NormPruner(model, cfg_list)
# pruner = L2NormPruner(model, cfg_list)
#pruner = FPGMPruner(model, cfg_list)
_, masks = pruner.compress()
pruner.show_pruned_weights()
pruner._unwrap_model()

m_speedup = ModelSpeedup(model, im, masks_file=masks)
m_speedup.speedup_model()


# retune model size for cspnet concat 
model.model[10].conv = nn.Conv2d(model.model[9].cv2.conv.out_channels, model.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.model[13].cv1.conv = nn.Conv2d(model.model[10].conv.out_channels + model.model[6].cv3.conv.out_channels, model.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.model[13].cv2.conv = nn.Conv2d(model.model[10].conv.out_channels + model.model[6].cv3.conv.out_channels, model.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.model[17].cv1.conv = nn.Conv2d(model.model[14].conv.out_channels + model.model[4].cv3.conv.out_channels, model.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.model[17].cv2.conv = nn.Conv2d(model.model[14].conv.out_channels + model.model[4].cv3.conv.out_channels, model.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.to(device=device)
model.eval()
_,__,___ = count_flops_params(model,im)

print(model)
torch.save(model,"./checkpoint/pruned_yolov5s.pt")


for _ in range(100):
    use_mask_out = model(im)

start = time.time()
for _ in range(100):
    use_mask_out = model(im)

pruned_time = (time.time() - start)*100

print('elapsed time_before_pruned: ', origin_time)
print('elapsed time when use mask: ', pruned_time)
torch.save(model,"./checkpoint/pruned_yolov5sb.pt")
'''