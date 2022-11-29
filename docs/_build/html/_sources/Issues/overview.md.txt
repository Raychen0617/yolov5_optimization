# Common Issues 

## Error: Yolov5 convert torch script error

```python 
File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/warnings.py", line 489
    def __exit__(self, *exc_info):
                       ~~~~~~~~~ <--- HERE
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
'__torch__.warnings.catch_warnings' is being compiled since it was called from 'SPPF.forward'
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/common.py", line 331
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
             ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
```

Fix: please refer to [here](https://github.com/ultralytics/yolov5/issues/1217) 



## Error: Model cannot to device, half(), float()

```python
File "/home/raytjchen/Desktop/code/yolov5_optimization/models/yolo.py", line 298, in _apply
    self = super()._apply(fn)
TypeError: super(type, obj): obj must be an instance or subtype of type
```

Fix: remove @model_wrapper from class Model (@model_wrapper need to be add when running NAS code)

## Error: Model cannot load old checkpoints 
```python
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/torch/serialization.py", line 712, in load
    return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/torch/serialization.py", line 1049, in _load
    result = unpickler.load()
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/torch/serialization.py", line 1042, in find_class
    return super().find_class(mod_name, name)
AttributeError: Can't get attribute 'ValueChoiceX' on <module 'nni.retiarii.nn.pytorch.api' from '/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/nni/retiarii/nn/pytorch/api.py'>
```

Fix: Downgrade nni to 2.8 to use old checkpoints 
```bash
conda activate python37
```

## Error: Model cannot load from checkpints 

Fix: Since there are several ways to store a pytorch model, there are also different ways to load a model
```python 
# Method1:
from nni.retiarii import fixed_arch
with fixed_arch("./output/Random_yolov5s.json"):
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device) 

# Method2:
model = torch.load(weights)['model'].cuda()
```

## Error: ModelNamespace is missing 
```bash
/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/nni/nas/utils/misc.py:188: RuntimeWarning: ModelNamespace is missing. You might have forgotten to use `@model_wrapper`. Some features might not work. This will be an error in future releases.
  'Some features might not work. This will be an error in future releases.', RuntimeWarning)
```
Fix: uncomment the @model_wrapper at [here](https://github.com/Raychen0617/yolov5_optimization/blob/master/models/yolo.py#L149)

## Error: Fixed context with {label} not found. Existing values are: {ret}
```bash 
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/yolo.py", line 177, in __init__
    self.model, self.save = nas_parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/yolo.py", line 765, in nas_parse_model
    m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/common.py", line 221, in __init__
    choice.append(NASC3sub(c1, c2, inputshape, id , n, shortcut, g, e, scale))
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/common.py", line 245, in __init__
    self.m = nn.Sequential(*(Bottleneck(round(scale * c_) , round(scale * c_) , shortcut, g, e=1.0) for _ in range(n)))
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/common.py", line 245, in <genexpr>
    self.m = nn.Sequential(*(Bottleneck(round(scale * c_) , round(scale * c_) , shortcut, g, e=1.0) for _ in range(n)))
  File "/home/raytjchen/Desktop/code/yolov5_optimization/models/common.py", line 138, in __init__
    scale = nn.ValueChoice([1, 1.5, 2])
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/typing.py", line 821, in __new__
    obj = super().__new__(cls, *args, **kwds)
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/nni/nas/nn/pytorch/mutation_utils.py", line 29, in __new__
    return cls.create_fixed_module(*args, **kwargs)
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/nni/nas/nn/pytorch/choice.py", line 870, in create_fixed_module
    value = get_fixed_value(label)
  File "/home/raytjchen/anaconda3/envs/tensorflow2/lib/python3.7/site-packages/nni/nas/nn/pytorch/mutation_utils.py", line 54, in get_fixed_value
    raise KeyError(f'Fixed context with {label} not found. Existing values are: {ret}'
```
Fix: The Nas output json file does not match the current NAS model keys. Change the output json or modify the NAS Model. 
