import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.util.train_util import DLContext, get_device, train
from torchvision import datasets, transforms
from tinynn.util.cifar10 import train_one_epoch, validate
from tinynn.converter import TFLiteConverter



normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=False),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

device = "cpu"
#model = timm.create_model('resnet50', pretrained=False)
#model.fc = torch.nn.Linear(model.fc.in_features, 10, bias=True)
#model = torch.load("./2022-07-26-15-38-02-057344/best_result/model.pth").cpu()


dummy_input = torch.rand(10, 3, 32, 32)
model = timm.create_model('resnet50', pretrained=False)
model.fc  = nn.Linear(model.fc.in_features, 10)
from nni.compression.pytorch import ModelSpeedup
ModelSpeedup(model, dummy_input, "./2022-07-27-13-06-54-431618/best_result/masks.pth").speedup_model()
model.load_state_dict(torch.load("./output/pruned.pth"))

# TinyNeuralNetwork provides a QATQuantizer class that may rewrite the graph for and perform model fusion for
# quantization. The model returned by the `quantize` function is ready for QAT.
# By default, the rewritten model (in the format of a single file) will be generated in the working directory.
# You may also pass some custom configuration items through the argument `config` in the following line. For
# example, if you have a QAT-ready model (e.g models in torchvision.models.quantization),
# then you may use the following line.
#   quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'rewrite_graph': False})
# Alternatively, if you have modified the generated model description file and want the quantizer to load it
# instead, then use the code below.
#     quantizer = QATQuantizer(
#         model, dummy_input, work_dir='out', config={'force_overwrite': False, 'is_input_quantized': None}
#     )
# The `is_input_quantized` in the previous line is a flag on the input tensors whether they are quantized or
# not, which can be None (False for all inputs) or a list of booleans that corresponds to the inputs.
# Also, we support multiple qschemes for quantization preparation. There are several common choices.
#   a. Asymmetric uint8. (default) config={'asymmetric': True, 'per_tensor': True}
#      The is the most common choice and also conforms to the legacy TFLite quantization spec.
#   b. Asymmetric int8. config={'asymmetric': True, 'per_tensor': False}
#      The conforms to the new TFLite quantization spec. In legacy TF versions, this is usually used in post
#      quantization. Compared with (a), it has support for per-channel quantization in supported kernels
#      (e.g Conv), while (a) does not.
#   c. Symmetric int8. config={'asymmetric': False, 'per_tensor': False}
#      The is same to (b) with no offsets, which may be used on some low-end embedded chips.
#   d. Symmetric uint8. config={'asymmetric': False, 'per_tensor': True}
#      The is same to (a) with no offsets. But it is rarely used, which just serves as a placeholder here.

quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'asymmetric': True, 'per_tensor': True})
qat_model = quantizer.quantize()


device = get_device()
qat_model.to(device=device)

# When adapting our framework to the existing training code, please make sure that the optimizer and the
# lr_scheduler of the model is redefined using the weights of the new model.
# e.g. If you use `get_optimizer` and `get_lr_scheduler` for constructing those objects, then you may write
#   optimizer = get_optimizer(qat_model)
#   lr_scheduler = get_lr_scheduler(optimizer)

context = DLContext()
context.device = device
context.train_loader, context.val_loader = train_loader, test_loader
context.max_epoch = 1
# !!! use the same criterion and optimizer when u are training
context.criterion = criterion = torch.nn.CrossEntropyLoss()
context.optimizer = torch.optim.SGD(qat_model.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
context.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

# Quantization-aware training
train(qat_model, context, train_one_epoch, validate, qat=True)

with torch.no_grad():
    
    qat_model.cpu()

    # The step below converts the model to an actual quantized model, which uses the quantized kernels.
    qat_model = torch.quantization.convert(qat_model)

    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = quantizer.backend

    # The code section below is used to convert the model to the TFLite format
    # If you need a quantized model with a specific data type (e.g. int8)
    # you may specify `quantize_target_type='int8'` in the following line.
    # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
    # you may specify `strict_symmetric_check=True` in the following line.
    converter = TFLiteConverter(qat_model, torch.rand(1, 3, 32, 32), tflite_path='./output/qat_model.tflite')
    converter.convert()