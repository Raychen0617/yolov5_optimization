from optimizer.prune import prune
from optimizer.match import match, extract_backbone
from models.yolo import BACKBONE
import torch 
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', type=str, default='./checkpoint/yolov5s.pt', help='yolo weights path')
    parser.add_argument('--save_path', type=str, default='./checkpoint/yolov5s.pt', help='yolo weights path')
    parser.add_argument('--sparsity', type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    
    device = "cuda:0"
    args = parse_args()

    if args.yolo[0:3] == "exp":
        args.yolo = "./runs/train/" + args.yolo + "/weights/best.pt"

    ori_backbone_yaml = './models/yolov5sb.yaml'
    yolo = torch.load(args.yolo)
    
    if type(yolo) is dict:
        yolo = yolo['model']

    backbone = extract_backbone(BACKBONE(cfg=ori_backbone_yaml, nc=200).backbone, yolo)
    backbone.to(device=device).float()
    backbone = prune(model=backbone, save=None, sparsity=float(args.sparsity), method="L2")
    yolo = match(yolo=yolo.float(),  pruned_yolo=backbone.float(), save=None)
    #print(yolo)
    torch.save(yolo, args.save_path)

