from turtle import forward
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
from models.yolo import Backbone, Model
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript
from torchvision.datasets import CIFAR100
from torchvision import transforms

# Create backbone 
class NASBACKBONE(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(cfg='./models/yolov5xb.yaml').to(device=device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1280, 100, bias=True)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


device = torch.device("cuda:0")
model = NASBACKBONE().to(device=device)

#INPUT
imgsz = (640, 640)
imgsz *= 2 if len(imgsz) == 1 else 1 # expand

gs = 32 # grid size (max stride)
imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

dataset_train = CIFAR100('../datasets', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.Resize((640,640)),
        transforms.ToTensor(),
        normalize,
]), download=True)


dataset_valid = CIFAR100('../datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640,640)),
        normalize,
]))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return 

def reward_accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

# use NAS here
def top1_accuracy(output, target):
    # this is the function that computes the reward, as required by ENAS algorithm
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

def metrics_fn(output, target):
    # metrics function receives output and target and computes a dict of metrics
    return {"acc1": reward_accuracy(output, target)}



from nni.algorithms.nas.pytorch.darts import DartsTrainer
trainer = DartsTrainer(model,
                        loss=criterion,
                        metrics=metrics_fn,
                        optimizer=optimizer,
                        num_epochs=1,
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        batch_size=16,
                        log_frequency=10,
                        unrolled=False)
trainer.enable_visualization()
trainer.train()  # training
trainer.export(file="./output/nas_yolov5xb.json")  # export the final architecture to file
print(model)
torch.save(model,"./checkpoint/nas_yolov5xb.pt")

'''
# ENAS
from nni.algorithms.nas.pytorch import enas
trainer = enas.EnasTrainer(model,
                           loss=criterion,
                           metrics=metrics_fn,
                           reward_function=top1_accuracy,
                           optimizer=optimizer,
                           batch_size=16,
                           num_epochs=2,  # 10 epochs
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           log_frequency=10,
                           workers=0
                        )
'''

