#from optimizer.prune import prune
from optimizer.match import match, fixed_nas, match_nas
#from vision_toolbox import backbones
from optimizer.model_evaluation import evaluate_model
from models.experimental import attempt_load
from models.yolo import Detect, Model, NASBACKBONE
import torch 
from utils.convert_weight import convert_weights_direct
#from optimizer.convert_compare import convert_and_compare
'''
device = "cuda:0"
from nni.retiarii import fixed_arch
with fixed_arch('./output/nas_yolov5sb.json'):
    model = NASBACKBONE(cfg="./models/yolov5sb_nas.yaml", nc=200).to(device=device)
    #torch.save(final_model, save_path)

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
model = match_nas(yolo=yolo, nas_backbone=model, save="./checkpoint/darts_yolov5s.pt")
model = torch.load("./checkpoint/darts_yolov5s.pt").cuda()

for k, m in model.named_modules():
    if isinstance(m, Detect):
        m.inplace = False
        m.export = True

convert_and_compare(model=model, output_path="./checkpoint/test.tflite", dummy_input=torch.rand(1,3,640,640))

nas_backbone = torch.load(weights)
pruned_nas_backbone = prune(model=nas_backbone, save=None,  sparsity=0.5, method="FPGM")

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
nas_pruned = match_nas(yolo=yolo, nas_backbone=nas_backbone, save="./checkpoint/darts_pruned_yolov5s.pt")

# Evaluating two models 
evaluate_model(model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False), dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=False)
print()
evaluate_model(model=nas_pruned, dummy_input=torch.rand(1,3,640,640), device=device,  testspeed=True, testflopsandparams=True)
'''

device = "cuda:0"
weights = "./checkpoint/darts_yolov5s.pt"
model = torch.load(weights)

ori_model='./models/yolov5s.yaml'
yolo = Model(ori_model).to(device=device)  
model = match_nas(yolo=yolo, nas_backbone=model, save="./checkpoint/enas_yolov5s.pt")
model = torch.load("./checkpoint/enas_yolov5s.pt").cuda()
