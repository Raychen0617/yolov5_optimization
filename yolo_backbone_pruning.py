from imageio import save
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
from models.yolo import Model
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript

device = torch.device("cuda:0")
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#model = attempt_load('yolov5s.pt', inplace=True, fuse=False, device=device) # load FP32 model
model = Model(cfg='./models/yolov5s.yaml').to(device=device)
model.eval()


dummy_input = torch.rand(1, 3, 640, 640).to(device=device)
print(model(dummy_input)[0].shape)


'''
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
'''

for k, m in model.named_modules():
    #print(k)
    if isinstance(m, Conv): # assign export-friendly activations
        #if isinstance(m.act, nn.SiLU):
        #    m.act = torch.nn.ReLU(inplace=False)
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
    #if hasattr(m, 'forward_export'):
    #    m.forward = m.forward_export # assign custom forward (optional)

imgsz = (640, 640)
imgsz *= 2 if len(imgsz) == 1 else 1 # expand

gs = int(max(model.stride)) # grid size (max stride)
imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection

cfg_list = [{
    'sparsity_per_layer': 0.1,
    'op_types': ['Conv2d'],
}, {
    'exclude': True,
    'op_names': ['model.24.m.0', 'model.24.m.1', 'model.24.m.2','model.13.cv1','model.13.cv2', 'model.13.cv3']
}]

# 1.
for _ in range(100):
    use_mask_out = model(im)
    
start = time.time()
for _ in range(100):
    use_mask_out = model(im)
print('elapsed time_before_pruned: ', (time.time() - start)*100)


pruner = L1NormPruner(model, cfg_list)
# pruner = L2NormPruner(model, cfg_list)
#pruner = FPGMPruner(model, cfg_list)
_, masks = pruner.compress()
#pruner.export_model(model_path='pruned_yolov5m.pt', mask_path='pruned_yolov5_mask.pt')
pruner.show_pruned_weights()
pruner._unwrap_model()
print("im.shape:",im.shape)

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

start = time.time()
for _ in range(100):
    use_mask_out = model(im)

dummy_input = torch.rand(1, 3, 640, 640).to(device=device)
print(model(dummy_input)[0].shape)


print('elapsed time when use mask: ', (time.time() - start)*100)