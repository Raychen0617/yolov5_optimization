# One-Shot NAS on YOLOv5 backbone


## Define NAS Model 
Defining a model is almost the same as defining a PyTorch. You need to replace the code import torch.nn as nn with import nni.retiarii.nn.pytorch as nn and add `@model_wrapper` at the beginning of the model 

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


### Construct YOLOv5 backbone that supports NAS componenets (NASC3, NASConv) 
[full_code](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolo.py#L359)
```python
class Backbone(nn.Module):
    parse_backbone(d, ch)
```
The function that supports parsing NAS components in backbone
[full_code](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolo.py#L520)
```python
def parse_backbone(d, ch):  # model_dict, input_channels(3)

    component_mapping = {"NASConv":NASConv, "NASC3":NASC3}

    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        
        if m in component_mapping.keys():
            m = component_mapping[m]

        if m in (NASConv, Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, NASC3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

            if m in [BottleneckCSP, NASC3, C3, C3TR, C3Ghost, C3x]:
                args.insert(2, n)  # number of repeats
                n = 1

            if m in [NASConv, NASC3]:
                args.insert(2,pre_shape)
                args.insert(3, i)
```

### Wrap the NAS backbone to a classification model for training 
```python
from nni.retiarii import model_wrapper

@model_wrapper
class NASBACKBONE(nn.Module):

    def __init__(self, cfg, nc):
        super().__init__()
        self.backbone = Backbone(cfg=cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if cfg[cfg.find("yolov5") + 6] == 'x':
            self.head = nn.Linear(1280, nc, bias=True)
        elif cfg[cfg.find("yolov5") + 6] == 's':
            self.head = nn.Linear(512, nc, bias=True)
        elif cfg[cfg.find("yolov5") + 6] == 'm':
            self.head = nn.Linear(768, nc, bias=True)
        elif cfg[cfg.find("yolov5") + 6] == 'n':
            self.head = nn.Linear(256, nc, bias=True)
        else:
            print("error loading models in backbone")
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
```

### Change Yaml File 

YoloV5's backbone NAS yaml [yolov5sb_nas.yaml](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolov5sb_nas.yaml)
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
nas_backbone_yaml = "./models/yolov5sb_nas.yaml"
model_space = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)
```

## Dataset
We use TinyImageNet for our classification training

```python
train_dataset, test_dataset, train_loader, test_loader = create_tinyimagenet(batchsize=1024)
```

## Evaluator

To begin exploring the model space, one firstly need to have an evaluator to provide the criterion of a “good model”.  The program is testing on classification tasks, so it can use `pl.Classification` as the evaluator


```python
import nni.retiarii.evaluator.pytorch.lightning as pl
evaluator = pl.Classification(
        # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
        # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
        train_dataloaders=pl.DataLoader(train_dataset, batch_size=512, num_workers=10),
        val_dataloaders=pl.DataLoader(test_dataset, batch_size=512, num_workers=10),
        # Other keyword arguments passed to pytorch_lightning.Trainer.
        max_epochs=1,
        gpus=1,
    )
```

## Startegy

In the experiment, we use DARTS and ENAS to explore our model space, for more one-shot strategies, please check out [here](https://nni.readthedocs.io/en/latest/nas/exploration_strategy.html)
```python
exploration_strategy = strategy.DARTS()
exploration_strategy = strategy.ENAS()
```

## Experiments
### Set configs 

```python
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], exploration_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'darts'
exp_config.execution_engine = 'oneshot'
exp_config.max_trial_number = 4  # spawn 4 trials at most
exp_config.trial_concurrency = 2  # will run two trials concurrently
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
```

### Launch experiment 
```python
port = 8081
exp.run(exp_config, port)
```

## Get the best usable model 

### Export the best backbone (model) 
```python
for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)

with open(save_json_path, 'w') as fp:
    json.dump(model_dict, fp)

with fixed_arch(save_json_path):
    backbone = NASBACKBONE(cfg=nas_backbone_yaml, nc=200).to(device=device)
```

### Map the backbone structure back to a YOLO model 

match_nas is a function that maps the backbone structure back and save the final detection model [full_code](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py#L47)
```python
save_model_path = "....."
yolo = Model(yolo_yaml).to(device=device) 
match_nas(yolo, backbone, save_model_path)
```


## Full Code On Github
[oneshot_nas.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/oneshot_nas.py)
