from optimizer.prune import prune
from optimizer.match import match, fixed_nas, match_nas
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, BACKBONE, Model, NASBACKBONE
import torch 
from utils.convert_weight import convert_weights_direct



device = "cuda:0"
weights = './checkpoint/darts_yolov5s.pt'
nas_backbone = torch.load(weights)
pruned_nas_backbone = prune(model=nas_backbone, save=None,  sparsity=0.25, method="FPGM")

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
nas_pruned = match_nas(yolo=yolo, nas_backbone=nas_backbone, save="./checkpoint/darts_pruned_yolov5s.pt")
