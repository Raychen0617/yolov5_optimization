from optimizer.prune import prune
from optimizer.match import match, fixed_nas, match_nas
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, BACKBONE, Model, NASBACKBONE
import torch 
from utils.convert_weight import convert_weights_direct


nas_yaml="./models/yolov5sb_nas.yaml"
nas_json = './output/nas_yolov5sb.json'
device = "cuda:0"
nas_backbone = NASBACKBONE(cfg=nas_yaml, nc=200).to(device=device).backbone.model

nas_backbone = fixed_nas(nas_backbone, nas_json)

weights = "./checkpoint/yolov5s_nas.pt"
nas = torch.load(weights)['model']

nas_backbone.load_state_dict(nas.model.state_dict(), strict=False)
nas_backbone = prune(model=nas_backbone, save="./checkpoint/nas_pruned_yolov5sb.pt",  sparsity=0.25, method="L1")

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
nas_pruned = match_nas( yolo=yolo, nas_json=None, nas_backbone=nas_backbone, save="./checkpoint/nas_pruned_yolov5s.pt", fixed=False)
print(nas_pruned)