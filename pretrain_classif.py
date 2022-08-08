# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported iterative pruning algorithms.
In this example, we show the end-to-end iterative pruning process: pre-training -> pruning -> fine-tuning.

'''
import string
from models.yolo import BACKBONE
import sys
import argparse
from tqdm import tqdm
import timm 
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
from nni.compression.pytorch.pruning import (
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner
)
from tinynn.converter import TFLiteConverter
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
import cv2
from nni.compression.pytorch import ModelSpeedup, apply_compression_results
from utils.dataloaders import create_tinyimagenet, create_cifar
from utils.convert_weight import convert_weights
from vision_toolbox import backbones


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(model, optimizer, criterion, epoch, train_loader):
    model.to(device)
    model.train()
    for data, target in tqdm(iterable=train_loader, desc='Epoch {}'.format(epoch)):     
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluator(model, test_loader):
    model.eval()
    total_acc1, total_acc5 = 0, 0
    with torch.no_grad():
        for data, target in tqdm(iterable=test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #pred = output.argmax(dim=1, keepdim=True)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1, acc5 = acc1.item(), acc5.item()   
            total_acc1 += acc1 * len(target)
            total_acc5 += acc5 * len(target)

            #correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_acc1 = total_acc1 / len(test_loader.dataset)
    total_acc5 = total_acc5 / len(test_loader.dataset)

    print('Acc1 =  {}%\n'.format(total_acc1))
    print('Acc5 = {}%\n'.format(total_acc5))
    return total_acc1, total_acc5

'''
class BACKBONE(nn.Module):

    def __init__(self, cfg, nc):
        super().__init__()
        self.backbone = backbones.darknet_yolov5s(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if cfg[cfg.find("yolov5") + 6] == 'x':
            self.head = nn.Linear(1280, nc, bias=True)
        elif cfg[cfg.find("yolov5") + 6] == 's':
            self.head = nn.Linear(512, nc, bias=True)
        elif cfg[cfg.find("yolov5") + 6] == 'm':
            self.head = nn.Linear(768, nc, bias=True)
        else:
            print("error loading models")
    
    def forward(self, x):
        #print(self.backbone)
        x = self.backbone(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
'''




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Iterative Example for model comporession')
    parser.add_argument('--pretrain-epochs', type=int, default=200,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--resume', type=bool, default=False,
                    help='resume or not')

    args = parser.parse_args()

    #model = BACKBONE(cfg='./models/yolov5sb.yaml',nc=200).to(device=device)
    #model = torch.load("./checkpoint/pruned_yolov5s.pt")
    #print(model)
    model = timm.create_model('resnet50', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 200, bias=True)
    #model.backbone.load_state_dict(convert_weights(backbones.darknet_yolov5s(pretrained=True)))

    '''    
    if args.resume:
        print("Loading checkpoints")
        model = torch.load('./checkpoint/tinyimagenet_yolov5mb.pt')
    else:
        model = BACKBONE(cfg='./models/yolov5mb.yaml',nc=200).to(device=device)
        #model.backbone.load_state_dict(torch.load('./checkpoint/yolov5m.pt')['model'].float().state_dict(), strict=False)
    '''
    
    _, _, train_loader, test_loader = create_tinyimagenet(batchsize=1024) #create_cifar("CIFAR100", batchsize=128) 

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=2e-5)
    criterion = torch.nn.CrossEntropyLoss()


    pre_best_acc1 = 37

    for i in range(args.pretrain_epochs):
        
        trainer(model, optimizer, criterion, i, train_loader)
        cur_acc1, cur_acc5 = evaluator(model, test_loader)
        if cur_acc1 > pre_best_acc1:
            print("Saving best model")
            torch.save(model, './checkpoint/tinyimagenet_yolov5sb.pt')
        pre_best_acc1 = max(pre_best_acc1, cur_acc1)  
    
    print("Best acc1", pre_best_acc1)

   
    
 