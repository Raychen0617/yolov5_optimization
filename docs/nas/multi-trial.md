# Multi-trial Detection NAS With Efficiency Rewards


## Define NAS Model 
Defining a model is almost the same as defining a PyTorch. You need to replace the code import torch.nn as nn with import nni.retiarii.nn.pytorch as nn and add `@model_wrapper` at the beginning of the model 

```python
from nni.retiarii import model_wrapper

@model_wrapper
class Model(nn.Module):
```

### Define Changable Modules

NASC3 is the variantion of the original CSP block (C3 module), it can adjust the output channel numnbers of cv1 and cv2. 
```python
import nni.retiarii.nn.pytorch as nn

class NASC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, inputshape=(), id=0, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # changeable output channels
        choice = []

        for scale in [1.0, 1.5, 2.0]:
            choice.append(NASC3sub(c1, c2, inputshape, id , n, shortcut, g, e, scale))
        self.total = LayerChoice(choice, label="c3_{}".format(id))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

NASConv is the varation of the original Conv module, it can adjust the kernel size and padding of convolutions and choose between different activations. 

```python
import nni.retiarii.nn.pytorch as nn

class NASConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, inputshape=(), id=0, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()

        choice = [nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)]
        # conv_2d_output_shape checks the output shape of convolutions (to make sure output size is the same)
        outputshape = conv_2d_output_shape(inputshape, k, s, autopad(k, p))
        for offsetk in (-2 , 2):
                for offsetpad in range(0 if p is None else -1*p,4):
                    if conv_2d_output_shape(inputshape, k+offsetk , s, autopad(k, p)+offsetpad) == outputshape:
                        choice.append(nn.Conv2d(c1, c2, k+offsetk, s, autopad(k, p)+offsetpad, groups=g, bias=False))

        self.conv  = LayerChoice(choice, label="nasconv_{}".format(id))
        self.shape = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        act_choice = [nn.SiLU(), nn.Identity(), nn.ReLU()]  # activation choices
        self.act = LayerChoice(act_choice, label="nasconv_{}_act".format(id))
```

### Change Yaml File 

YoloV5s NAS model's yaml (only backbone part) [full yolov5s_nas.yaml](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolov5s_nas.yaml)
```yaml
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, NASConv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, NASConv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, NASC3, [128]],
   [-1, 1, NASConv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, NASC3, [256]],
   [-1, 1, NASConv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, NASC3, [512]],
   [-1, 1, NASConv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, NASC3, [1024]],
   [-1, 1, SPPF, [1024, 5]]
  ]
```

### Setup User-defined Nas Model 
```python
device = "cuda:0"
hyp = "data/hyps/hyp.scratch-low.yaml" # hyper-parameters in yolov5
cfg="./models/yolov5s_nas.yaml" # yaml file
model_space = Model(cfg=cfg, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
```

## Explore The Defined Model Space


### Pick An Exploration Strategy

NNI supports many [exploration startegies](https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html), simply choosing (i.e., instantiate) an exploration strategy as below.<br>

```python
import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted
```

### Customize A Model Evaluator

Setup parameters (ex: batch size, epochs) for model 

```python

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
    batch_size = 64
    single_cls = False
    from utils.general import colorstr
    train_path = "/home/raytjchen/Desktop/code/datasets/coco128/images/train2017"
    gs = 32
    nbs = 64  # nominal batch size
    epochs = 20 # how many epochs to train for a single choice 
```

Create optimizer and scheduler for Yolov5

```python
    # Optimizer
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    from utils.torch_utils import smart_optimizer
    optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
```

Create trainloader and dataloader on coco dataset 

```python
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
```

Specify Model's attribute 

```python
    # Model attributes
    hyp['obj'] *= (imgsz / 640) ** 2  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = 80  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    from utils.general import labels_to_class_weights
    model.class_weights = labels_to_class_weights(dataset.labels, 80).to(device) * 80  # attach class weights
    model.names = "nas_yolov5s"
```

Start training 
```python

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
```

### Create The Evaluator 
```python
fmrom nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)
```

## Launch An Experiment 

After all the above are prepared, it is time to start an experiment to do the model search. An example is shown below.

```python
from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'yolov5s_nas_search'
```
The following configurations are useful to control how many trials to run at most / at the same time.
```python
exp_config.max_trial_number = 4   # spawn 4 trials at most
exp_config.trial_concurrency = 1  # will run two trials concurrently 
```
Remember to set the following config if you want to GPU. use_active_gpu should be set true if you wish to use an occupied GPU (possibly running a GUI).
```python
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
```

Launch the experiment
```python
exp.run(exp_config, 8083)
```

## Export Best Model
``` python
for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)
save_json_path = "./yolov5s_nas.json"
with open(save_json_path, 'w') as fp:
    json.dump(model_dict, fp)
```

## Full Code On Github
[hello_nas.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/hello_nas.py)