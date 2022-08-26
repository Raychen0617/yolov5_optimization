from optimizer.prune import prune
from optimizer.match import match, fixed_nas, match_nas
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, BACKBONE, Model, NASBACKBONE
import torch 
from utils.convert_weight import convert_weights_direct



device = "cpu"
weights = './checkpoint/backbone/enas_yolov5sb.pt'
nas_backbone = torch.load(weights)
#pruned_nas_backbone = prune(model=nas_backbone, save=None,  sparsity=0.25, method="FPGM")

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
nas_pruned = match_nas(yolo=yolo, nas_backbone=nas_backbone, save="./checkpoint/enas.pt")
nas_pruned.cuda()


# Evaluating two models 
evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)
print()
evaluate_model(model=nas_pruned, dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)
