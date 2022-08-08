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
from models.yolo import NASBACKBONE, Backbone, Model
from models.common import *
from models.experimental import attempt_load
from export import export_torchscript
from torchvision.datasets import CIFAR100
from torchvision import transforms
from utils.dataloaders import create_tinyimagenet

device = torch.device("cuda:0")
model = NASBACKBONE(cfg="yolov5sb_nas.yaml", nc=200).to(device=device)

dataset_train, dataset_valid, train_loader, test_loader = create_tinyimagenet(batchsize=1024)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=2e-5)
criterion = torch.nn.CrossEntropyLoss()


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
trainer.export(file="./output/nas_yolov5sb.json")  # export the final architecture to file
torch.save(model,"./checkpoint/nas_yolov5sb.pt")

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

