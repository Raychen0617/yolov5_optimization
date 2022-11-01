# Prune Yolov5 backbone 



## Preparations

## Create Backbone 
```python
ori_backbone_model = './models/yolov5sb.yaml'
model = BACKBONE(cfg=ori_backbone_model, nc=200).to(device=device)
```

## Load Pretrained Backbone 
The pretrained backbone is referenced from vision_toolbox, which is trained on ImageNet. For more information, please refer to [vision_toolbox](https://github.com/gau-nernst/vision-toolbox)

```python 
from vision_toolbox import backbones
model.backbone.load_state_dict(convert_weights_direct(pretrain_backbone))
```

## Pruning

### Pruning Function 

Full code: [prune.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/prune.py)

Change the model to make it export-friendly
```python
for k, m in model.named_modules(): 
    if isinstance(m, Conv): # assign export-friendly activations
        if isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
```

Setup model configs
```python
imgsz = (64, 64)
imgsz *= 2 if len(imgsz) == 1 else 1 # expand
gs = 32 # grid size (max stride)
imgsz = [check_img_size(x, gs) for x in imgsz] # verify img_size are gs-multiples
im = torch.zeros(1, 3, *imgsz).to(device) # image size(1,3,320,192) BCHW iDetection
```

Setup pruning configs. The following cfg_list means all layers whose type is Conv2d will be pruned. The final sparsity ratio for each layer is defined by variable sparsity. For more cfg_list settings, please refer to [compression config specification](https://nni.readthedocs.io/en/latest/compression/compression_config_list.html)
```python
cfg_list = [{
    'sparsity_per_layer': sparsity,
    'op_types': ['Conv2d'],
}]
```

There are many pruners supported by NNI, for more information, please refer to [NNI](https://nni.readthedocs.io/en/latest/compression/pruner.html)
```python
if method == "L1":
    pruner = L1NormPruner(model, cfg_list)
elif method == "L2":
    pruner = L2NormPruner(model, cfg_list)
elif method == "FPGM":
    pruner = FPGMPruner(model, cfg_list)
else:
    print("Method is not supported !!! (prune.py)")
    return 
```

Generate masks for each pruned layers
```python
_, masks = pruner.compress()
pruner.show_pruned_weights()
pruner._unwrap_model()
```

Masks can be used to check model performance of a specific pruning (or sparsity), but there is no real speedup. Therefore, after generating the masks, we have to replace our layers with smaller layers without masks for real speedup. 

```python
m_speedup = ModelSpeedup(model, im, masks_file=masks)
m_speedup.speedup_model()
```

### Set Prune Configs and Prune Backbone 
```python 
save_prune = "./checkpoint/test_pruned_yolov5sb.pt"
sparsity=0.25
method="L1"
model = prune(model=model, save=save_prune, sparsity=sparsity, method=method)
```

## Map pruned backbone to a Yolov5

### Match function 

Full code [match.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/optimizer/match.py)

```python
pruned_yolo_layer = {}
    for name, model_type in pruned_yolo.named_modules():
        if isinstance(model_type, NASConv) or isinstance(model_type, NASC3) or isinstance(model_type, Conv):
            pruned_yolo_layer[name] = model_type

    for name, model_type in yolo.named_modules():
        if isinstance(model_type, Conv):
            
            if pruned_yolo_layer.get(name) is None:
                print(name, "cannot be found")

            else:
                #print(name, pruned_yolo_layer[name], model_type)
                if pruned_yolo_layer[name].conv.in_channels != model_type.conv.in_channels  or pruned_yolo_layer[name].conv.out_channels != model_type.conv.out_channels \
                    or pruned_yolo_layer[name].conv.kernel_size != model_type.conv.kernel_size  or pruned_yolo_layer[name].conv.padding != model_type.conv.padding:
                    model_type.conv = pruned_yolo_layer[name].conv
                    
                if pruned_yolo_layer[name].bn != model_type.bn:
                    model_type.bn = pruned_yolo_layer[name].bn
                
                if pruned_yolo_layer[name].act != model_type.act:
                    model_type.act = pruned_yolo_layer[name].act
            
    # retune model size for cspnet concat 
    yolo.model[9].cv1.conv = nn.Conv2d(yolo.model[8].cv3.conv.out_channels, yolo.model[9].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[10].conv = nn.Conv2d(yolo.model[9].cv2.conv.out_channels, yolo.model[10].conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv1.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[13].cv2.conv = nn.Conv2d(yolo.model[10].conv.out_channels + yolo.model[6].cv3.conv.out_channels, yolo.model[13].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv1.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv1.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    yolo.model[17].cv2.conv = nn.Conv2d(yolo.model[14].conv.out_channels + yolo.model[4].cv3.conv.out_channels, yolo.model[17].cv2.conv.out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)

```

### Call match function 
```python
yolo = Model(ori_model).to(device=device)  
model = match(yolo=yolo,  pruned_yolo=model.backbone, save=save_matched_prune)
```

## Full Code On Github
[pruning.py](https://github.com/Raychen0617/yolov5_optimization/blob/master/pruning.py)
