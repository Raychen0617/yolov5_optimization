from optimizer.prune import prune
from optimizer.match import match
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load

import torch 
'''
ori_model = './models/yolov5s.yaml'
ori_backbone_model = './models/yolov5sb.yaml'
save_prune = "./checkpoint/pruned_yolov5sb.pt"
save_matched_prune = "./checkpoint/pruned_yolov5s.pt"
pretrain_backbone=backbones.darknet_yolov5s(pretrained=True)
device = "cuda:0"

prune(ori_model=ori_backbone_model, pretrain_backbone=pretrain_backbone, save=save_prune, sparsity=0.25, method="FPGM")
match(ori_model=ori_model, pruned_model=save_prune, save=save_matched_prune)

evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device)
print()
evaluate_model(model=torch.load(save_matched_prune), dummy_input=torch.rand(1,3,640,640), device=device)
'''

model = torch.load('./checkpoint/yolov5s_nas.pt')['model']
device = "cpu"

evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device, testspeed=True, testflopsandparams=False)
print()
evaluate_model(model=model.float(), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)