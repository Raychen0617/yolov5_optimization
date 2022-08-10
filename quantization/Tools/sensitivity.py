from pyexpat import model
from nni.compression.pytorch.utils.sensitivity_analysis import SensitivityAnalysis
import torch 
import os 
import timm
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def val(model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batchid, (data, label) in enumerate(test_loader):
            data, label = data.cuda(), label.cuda()
            out = model(data)
            _, predicted = out.max(1)
            total += data.size(0)
            correct += predicted.eq(label).sum().item()
    print(correct/total)
    return correct / total

def trainer(model, optimizer, criterion, epoch):
    model.train()
    for data, target in tqdm(iterable=train_loader, desc='Epoch {}'.format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



model = timm.create_model('resnet18', pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10, bias=True)
model = model.cuda()
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for i in range(20):
    trainer(model, optimizer, criterion, i)
    val(model=model)
val(model=model)
s_analyzer = SensitivityAnalysis(model=model, val_func=val, sparsities=[0.8])

specified_layers = []

for i in range(1, 5):
    for j in range(2):
        specified_layers.append("layer"+str(i)+"."+str(j)+".conv1")
        specified_layers.append("layer"+str(i)+"."+str(j)+".conv2")


sensitivity = s_analyzer.analysis(val_args=[model], specified_layers=specified_layers)
outdir = "./output"
filename = "sensitivity"

s_analyzer.export(os.path.join(outdir, filename))