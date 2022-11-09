#!/bin/sh

python iterative_pruning.py --yolo "./checkpoint/multi-trail_yolov5s.pt" --save_path "./iterative_pruning/yolo.pt"
python train.py --data coco.yaml --weights "./iterative_pruning/yolo.pt" --img 640  --epochs 5

new_weight=$(ls ./runs/train/ | tail -n 1)
python iterative_pruning.py --yolo $new_weight --save_path "./iterative_pruning/yolo.pt"
python train.py --data coco.yaml --weights "./iterative_pruning/yolo.pt" --img 640  --epochs 5

new_weight=$(ls ./runs/train/ | tail -n 1)
python iterative_pruning.py --yolo $new_weight --save_path "./iterative_pruning/yolo.pt"
python train.py --data coco.yaml --weights "./iterative_pruning/yolo.pt" --img 640  --epochs 5

new_weight=$(ls ./runs/train/ | tail -n 1)
python iterative_pruning.py --yolo $new_weight --save_path "./iterative_pruning/yolo.pt"
python train.py --data coco.yaml --weights "./iterative_pruning/yolo.pt" --img 640  --epochs 86
