
from ast import Nonlocal
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from utils.dataloaders import create_tinyimagenet
from models.yolo import NASBACKBONE, Model
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
import json 
from optimizer.match import match_nas
from optimizer.prune import prune
from nni.retiarii import fixed_arch

device = torch.device("cuda:0")

########################    USP    ######################################
save_model_path = "./checkpoint/dartsv2_yolov5s.pt"
yolo_yaml= "./models/yolov5s.yaml"
save_json_path = "./output/dartsv2_yolov5s.json"
nas_backbone_yaml = "./models/yolov5sb_nas.yaml"

# Pruning configs
to_prune = False
sparsity = 0.25
method = "FPGM"
save_pruned_backbone = None
########################    USP    ######################################


model_space = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)
train_dataset, test_dataset, train_loader, test_loader = create_tinyimagenet(batchsize=1024)


evaluator = pl.Classification(
    # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
    # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
    train_dataloaders=pl.DataLoader(train_dataset, batch_size=512, num_workers=10),
    val_dataloaders=pl.DataLoader(test_dataset, batch_size=512, num_workers=10),
    # Other keyword arguments passed to pytorch_lightning.Trainer.
    max_epochs=100,
    gpus=1,
)

########################    NAS algorithm   ######################################
#exploration_strategy = strategy.ENAS(reward_metric_name='val_acc')
exploration_strategy = strategy.DARTS()

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'darts'
exp_config.execution_engine = 'oneshot'


# The following configurations are useful to control how many trials to run at most / at the same time.
exp_config.max_trial_number = 4  # spawn 4 trials at most
exp_config.trial_concurrency = 2  # will run two trials concurrently
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True


# Launch the experiment
exp.run(exp_config, 8081)

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)

with open(save_json_path, 'w') as fp:
    json.dump(model_dict, fp)


with fixed_arch(save_json_path):
    backbone = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)

if to_prune:
    backbone = prune(model=backbone, save=save_pruned_backbone, sparsity=sparsity, method=method)


yolo = Model(yolo_yaml).to(device=device) 
match_nas(yolo, backbone, save_model_path)

print("Success, json file is saved at ", save_json_path,"    pt file is saved at", save_model_path)
print("You can train the model by runining     --python train.py --weights ", save_model_path, " --data coco.yaml --epochs 101")