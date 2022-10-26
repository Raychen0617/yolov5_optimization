from turtle import back
from optimizer.prune import prune
from optimizer.match import match, fix_nasbackbone, fix_nasyolo
from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, BACKBONE, Model
import torch 
from utils.convert_weight import convert_weights_direct
import train
from models.yolo import NASBACKBONE,Model

device = "cuda:0"
nas_backbone_yaml = "./models/yolov5sb_nas.yaml"
nas_backbone_json = "./output/Random_yolov5s.json"
nas_full_model_weight = "./runs/train/exp2/weights/best.pt"
save_matched_prune = "./checkpoint/hello_nas_test.pt"

#train.run(data='coco128.yaml', imgsz=320, cfg='./models/yolov5s.yaml', epochs=1)
#exit()

# Backbone
from nni.retiarii import fixed_arch
with fixed_arch(nas_backbone_json):
    nasbackbone = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).backbone
#nasbackbone.load_state_dict(torch.load(nas_full_model_weight)['model'].state_dict(), strict=False)
ori_backbone_yaml = './models/yolov5sb.yaml'
backbone = BACKBONE(cfg=ori_backbone_yaml, nc=200).to(device=device).backbone
backbone = fix_nasbackbone(backbone, nasbackbone)


# Yolo
ori_yolo_yaml = './models/yolov5s.yaml'
yolo = Model(ori_yolo_yaml).to(device=device) 
nas_yolo = torch.load(nas_full_model_weight)['model']
yolo = fix_nasyolo(yolo=yolo, nas_yolo=nas_yolo, backbone_layer=8)
#evaluate_model(model=yolo.float(), dummy_input=torch.rand(1,3,640,640).float(), device=device,  testspeed=False, testflopsandparams=True)


# iterative pruning
iterations = 3

for iter in range(iterations):
    backbone.to(device=device)
    backbone = prune(model=backbone, save=None, sparsity=0.1, method="L2")
    yolo = match(yolo=yolo,  pruned_yolo=backbone, save=None)
    yolo.to(device=device)
    # train & eval
    opt, yolo = train.run(data='coco128.yaml', imgsz=640, cfg='./models/yolov5s.yaml', run_model=yolo, epochs=1)
    
    # load yolo state dict back to backbone
    backbone.load_state_dict(yolo.state_dict(), strict=False)


print(yolo)
save_final_model_path = "./checkpoint/multi_trail_nas_ipruning.pt"
torch.save(yolo, save_final_model_path)