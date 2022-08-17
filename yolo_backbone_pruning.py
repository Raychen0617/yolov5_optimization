from optimizer.prune import prune
from optimizer.match import match
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, BACKBONE, Model
import torch 
from utils.convert_weight import convert_weights_direct


ori_model = './models/yolov5s.yaml'
ori_backbone_model = './models/yolov5sb.yaml'
save_prune = "./checkpoint/test_pruned_yolov5sb.pt"
save_matched_prune = "./checkpoint/test_pruned_yolov5s.pt"
pretrain_backbone=backbones.darknet_yolov5s(pretrained=True)
device = "cuda:0"

model = BACKBONE(cfg=ori_backbone_model, nc=200).to(device=device)
model.backbone.load_state_dict(convert_weights_direct(pretrain_backbone))
model = prune(model=model, save=save_prune, sparsity=0.25, method="L1")


yolo = Model(ori_model).to(device=device)  
model = match(yolo=yolo,  pruned_yolo=model.backbone, save=save_matched_prune)

# Evaluating two models 
evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)
print()
evaluate_model(model=torch.load(save_matched_prune), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)

'''
model = torch.load('./checkpoint/yolov5s_nas.pt')['model']
device = "cpu"
evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device, testspeed=True, testflopsandparams=False)
print()
evaluate_model(model=model.float(), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)
'''