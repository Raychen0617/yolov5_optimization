import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from optimizer.prune import prune
import os

from utils.general import check_dataset

########################    USP    ######################################
save_model_path = "./checkpoint/Random_yolov5s.pt"
yolo_yaml= "./models/yolov5s.yaml"
save_json_path = "./output/test_yolov5s.json"
nas_backbone_yaml = "./models/yolov5sb_nas.yaml"
nas_traing = True

# Pruning configs
to_prune = False
sparsity = 0.25
method = "L1"
save_pruned_backbone = None
########################    USP    ######################################

# Construct NAS Model 

from models.yolo import NASBACKBONE,Model
device = "cuda:0"
hyp = "data/hyps/hyp.scratch-low.yaml"
cfg="./models/yolov5s_nas.yaml"

import yaml
if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  

model_space = Model(cfg=cfg, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)

import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted

import nni

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils.dataloaders import create_tinyimagenet

'''
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
'''

def train_epoch(model, device, train_loader, optimizer, epoch):
    
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    model.train()
    hyp = "data/hyps/hyp.scratch-low.yaml"
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict


    from utils.loss import NASComputeLoss
    compute_loss = NASComputeLoss(model=model, h=hyp)
    nb = len(train_loader)
    #maps = np.zeros(80)  # mAP per class
    #results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    mloss = torch.zeros(3, device=device)
    optimizer.zero_grad()
    pbar = enumerate(train_loader)
    from tqdm import tqdm
    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    last_opt_step = -1
    
    for batch_idx, (imgs, targets, paths, _) in pbar:

        ni = batch_idx + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        pred = model(imgs)
        loss, loss_items = compute_loss(pred, targets.to(device))
        scaler.scale(loss).backward()
        
        # Optimizer
        if ni - last_opt_step >= accumulate:
            optimizer.step()


        mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)

        #pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (f'{epoch}/{3 - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
        
        if batch_idx % 10 == 0:
          print("box   obj  cls ")
          print(mloss)
    


def evaluate_model(model_detect):
    
    model = model_detect()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    # Parameters
    hyp = 'data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f) 
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    imgsz = 640
    batch_size = 16
    single_cls = False
    from utils.general import colorstr
    train_path = "/home/raytjchen/Desktop/code/datasets/coco128/images/train2017"
    gs = 32
    nbs = 64  # nominal batch size
    epochs = 20 # how many epochs to train for a single choice 

    # Optimizer
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    from utils.torch_utils import smart_optimizer
    optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Create Trainloader
    from utils.dataloaders import create_dataloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None,
                                              rect=False,
                                              rank=-1,
                                              workers=0,
                                              image_weights=False,
                                              quad=False,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
        
    # Testloader
    val_path = "/home/raytjchen/Desktop/code/datasets/coco128/images/train2017"
    val_loader = create_dataloader(
                                val_path,
                                imgsz,
                                batch_size // WORLD_SIZE * 2,
                                gs,
                                single_cls,
                                hyp=hyp,
                                cache=None,
                                rect=True,
                                rank=-1,
                                workers=0,
                                pad=0.5,
                                prefix=colorstr('val: '))[0]

    # Model attributes
    hyp['obj'] *= (imgsz / 640) ** 2  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = 80  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    from utils.general import labels_to_class_weights
    model.class_weights = labels_to_class_weights(dataset.labels, 80).to(device) * 80  # attach class weights
    model.names = "nas_yolov5s"

    # Start training

    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    import numpy as np
    nc = 80
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    from utils.loss import NASComputeLoss
    compute_loss = NASComputeLoss(model=model, h=hyp)
    
    for epoch in range(epochs):
        
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = enumerate(train_loader)
        from tqdm import tqdm
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
       
        optimizer.zero_grad()

        for batch_idx, (imgs, targets, paths, _) in pbar:
            
            ni = batch_idx + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device))
            scaler.scale(loss).backward()
            
            # Optimizer step on 
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni

            mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses
            
            if batch_idx % 10 == 0:
                print(mloss)
            
            # end batch ------------------------------------------------------------------------------------------------------------------------
        
        # Scheduler step
        scheduler.step()

        # Validate
        data_dict = check_dataset('data/coco128.yaml')
        import val as validate
        results, maps, _ = validate.run(data_dict,
                                batch_size=batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                half=False,
                                model=model,
                                single_cls=single_cls,
                                dataloader=val_loader,
                                save_dir="./output/",
                                plots=False,
                                #callbacks=callbacks,
                                compute_loss=compute_loss)

        print("result = ", results[:4])
        nni.report_intermediate_result(int(results[3]) * 1000)

    # report final test result
    nni.report_final_result(results[3] * 1000)


# %%
# Create the evaluator

from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'yolov5s_nas_search'

# %%
# The following configurations are useful to control how many trials to run at most / at the same time.

exp_config.max_trial_number = 4   # spawn 4 trials at most
exp_config.trial_concurrency = 1  # will run two trials concurrently 

# %%
# Remember to set the following config if you want to GPU.
# ``use_active_gpu`` should be set true if you wish to use an occupied GPU (possibly running a GUI).

exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True

# %%
# Launch the experiment. The experiment should take several minutes to finish on a workstation with 2 GPUs.

exp.run(exp_config, 8083)


# Visualize the Experiment
# ------------------------
#
# Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment.
# For example, open ``localhost:8081`` in your browser, 8081 is the port that you set in ``exp.run``.
# Please refer to :doc:`here </experiment/web_portal/web_portal>` for details.
#
# We support visualizing models with 3rd-party visualization engines (like `Netron <https://netron.app/>`__).
# This can be used by clicking ``Visualization`` in detail panel for each trial.
# Note that current visualization is based on `onnx <https://onnx.ai/>`__ ,
# thus visualization is not feasible if the model cannot be exported into onnx.
#
# Built-in evaluators (e.g., Classification) will automatically export the model into a file.
# For your own evaluator, you need to save your file into ``$NNI_OUTPUT_DIR/model.onnx`` to make this work.
# For instance,

import os
from pathlib import Path

def evaluate_model_with_visualization(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model_cls)


import json
for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)

input('press any button to exit')

'''
with open(save_json_path, 'w') as fp:
    json.dump(model_dict, fp)


device = "cuda:0"
from nni.retiarii import fixed_arch
with fixed_arch(save_json_path):
    backbone = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)

if to_prune:
    backbone = prune(model=backbone, save=save_pruned_backbone, sparsity=sparsity, method=method)


from optimizer.match import match_nas
yolo = Model(yolo_yaml).to(device=device) 
match_nas(yolo, backbone, save_model_path)

print("Success, json file is saved at ", save_json_path,"    pt file is saved at", save_model_path)
print("You can train the model by runining     python train.py --weights ", save_model_path, " --data coco.yaml --epochs 101")


'''