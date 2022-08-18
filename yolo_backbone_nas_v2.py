
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
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
from optimizer.match import match_nas
from models.yolo import Backbone, Model, NASBACKBONE



def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy



import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl


device = torch.device("cuda:0")
model_space = NASBACKBONE(cfg="./models/yolov5sb_nas.yaml", nc=200).to(device=device)
train_dataset, test_dataset, train_loader, test_loader = create_tinyimagenet(batchsize=1024)

from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = pl.Classification(
    # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
    # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
    train_dataloaders=pl.DataLoader(train_dataset, batch_size=256),
    val_dataloaders=pl.DataLoader(test_dataset, batch_size=256),
    # Other keyword arguments passed to pytorch_lightning.Trainer.
    max_epochs=10,
    gpus=1,
)
exploration_strategy = strategy.ENAS()


from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'test'
exp_config.execution_engine = 'oneshot'


# The following configurations are useful to control how many trials to run at most / at the same time.

exp_config.max_trial_number = 100   # spawn 4 trials at most
exp_config.trial_concurrency = 2  # will run two trials concurrently

# Remember to set the following config if you want to GPU.
# ``use_active_gpu`` should be set true if you wish to use an occupied GPU (possibly running a GUI).

exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True

# Launch the experiment. The experiment should take several minutes to finish on a workstation with 2 GPUs.

exp.run(exp_config, 8081)


import os
from pathlib import Path

# Relaunch the experiment, and a button is shown on Web portal.
#
# .. image:: ../../img/netron_entrance_webui.png
#
# Export Top Models
# -----------------
#
# Users can export top models after the exploration is done using ``export_top_models``.

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)


# Save the model 
from nni.retiarii import fixed_arch
save_path = "./checkpoint/darts_nasv2_yolov5s.pt"
temp_model = NASBACKBONE(cfg="./models/yolov5sb_nas.yaml", nc=200).to(device=device)
with fixed_arch(model_dict):
    final_model = NASBACKBONE(cfg="./models/yolov5sb_nas.yaml", nc=200).to(device=device)
    print(final_model(torch.rand(1,3,640,640).to(device=device)).shape == temp_model(torch.rand(1,3,640,640).to(device=device)).shape)
    torch.save(final_model, save_path)

# The output is ``json`` object which records the mutation actions of the top model.
# If users want to output source code of the top model,
# they can use :ref:`graph-based execution engine <graph-based-execution-engine>` for the experiment,
# by simply adding the following two lines.

exp_config.execution_engine = 'base'
export_formatter = 'code'