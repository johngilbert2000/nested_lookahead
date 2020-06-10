# Imports
import torch
import torch.optim as optim
import torchvision as tv

from NestedLookahead import NestedLookahead

import os
import time
import datetime
import pandas as pd
import argparse
import gc

gc.collect() # run garbage collection just in case

# Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_path = './cifar_data'
verbose = True
loss_func = torch.nn.CrossEntropyLoss()

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="number of epochs", type=int, default=30)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--bs", help="batch size", type=int, default=64)
parser.add_argument("--mom", help="momentum", type=float, default=0.9)
parser.add_argument("--wd", help="weight decay", type=float, default=0)
parser.add_argument("--opt", help="optimizer type", type=str, default='Adam', 
    choices=['SGD', 'Adam','Lookahead_SGD','Lookahead_Adam','NestedLookahead_SGD','NestedLookahead_Adam'])
parser.add_argument("--k", help="inner slow steps (nested lookahead)", type=int, default=5)
parser.add_argument("--a", help="inner slow step size (nested lookahead)", type=float, default=0.5)
parser.add_argument("--s", help="outer slow steps (nested lookahead)", type=int, default=5)
parser.add_argument("--h", help="outer slow step size (nested lookahead)", type=float, default=0.5)
parser.add_argument("--pullback", help="pullback type (nested lookahead)", type=str, default="None", 
    choices=["None", "reset", "pullback", "reset_inner", "reset_outer", "pullback_inner", "pullback_outer"])
parser.add_argument("--tag", help="tag for result csv", default=0)
#parser.add_argument("--verbose", help="prints loss and accuracy", default=True)
args = parser.parse_args()

# Parameters
#epochs = 5 # 30
#bs = 64 # 128
#lr = 0.001
#momentum = 0.9
#weight_decay = 1e-3
#opt_t = 'Adam' # optimizer type

# Parameters
epochs = args.epochs
bs = args.bs
lr = args.lr
momentum = args.mom
weight_decay = args.wd
opt_t = args.opt
tag = args.tag

assert opt_t in ['SGD', 'Adam','Lookahead_SGD','Lookahead_Adam','NestedLookahead_SGD','NestedLookahead_Adam']

# (Nested) Lookahead
# k=5 # inner slow steps
# a=0.5 # alpha; inner slow step size
# s=5 # outer slow steps
# h=0.5 # outer slow step size
# pullback=None

# (Nested) Lookahead Parameters
k = args.k # inner slow steps
a = args.a # alpha; inner slow step size
s = args.s # outer slow steps
h = args.h # outer slow step size
pullback = None if 'None' in args.pullback else args.pullback

# If not nested, only use inner slow steps and fast steps (outer slow steps = 0)
if (opt_t == 'Lookahead_SGD') or (opt_t == 'Lookahead_Adam'):
    s = 0

def to_device(*stuff):
    "Move stuff to device (gpu)"
    return [obj.to(device) for obj in stuff]
    
def to_cpu(*stuff):
    "Move stuff to cpu"
    return [obj.to('cpu') for obj in stuff]

# Data Transformations
trans = tv.transforms.Compose([
    tv.transforms.RandomCrop(32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(brightness=0, contrast=(0.7,1.), saturation=(0.7,1.), hue=(0.,0.1)),
    tv.transforms.RandomRotation(45),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and std taken from imagenet defaults
#     tv.transforms.RandomErasing(p=0.5, scale=(0.02,0.10), value=128)
        ])

# Datasets and Dataloaders
train_ds = tv.datasets.CIFAR10(root=root_path, train=True, download=True, transform=trans)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)

test_ds = tv.datasets.CIFAR10(root=root_path, train=False, download=True, transform=trans)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)

# Model
model = tv.models.resnet18(pretrained=False)
model = model.to(device)

# Set optimizer
if opt_t == 'SGD':
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

elif opt_t == 'Adam':
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

elif opt_t == 'Lookahead_SGD':
    inner_opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt = NestedLookahead(inner_opt, k=k, a=a, s=s, h=h, pullback=pullback)

elif opt_t == 'Lookahead_Adam':
    inner_opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt = NestedLookahead(inner_opt, k=k, a=a, s=s, h=h, pullback=pullback)

elif opt_t == 'NestedLookahead_SGD':
    inner_opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt = NestedLookahead(inner_opt, k=k, a=a, s=s, h=h, pullback=pullback)

elif opt_t == 'NestedLookahead_Adam':
    inner_opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt = NestedLookahead(inner_opt, k=k, a=a, s=s, h=h, pullback=pullback)

# Training Loop
def train(model, train_dl, loss_func, opt):
    model.train()
    correct = total = train_loss = 0
    accs = []
    for X, Y_true in train_dl:
        X, Y_true = to_device(X, Y_true)
        opt.zero_grad()
        Y_pred = model(X)
        loss = loss_func(Y_pred, Y_true)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        X, Y_true, Y_pred = to_cpu(X, Y_true, Y_pred)
        _, Y_pred = Y_pred.max(dim=1)
        correct += (Y_pred == Y_true).type(torch.IntTensor).sum().item()
        total += len(Y_true)
        accs.append(correct/total)
    return accs

# Validation / Test Loop
def validate(model, test_dl, loss_func):
    model.eval()
    correct = total = test_loss = 0
    accs = []
    with torch.no_grad():
        for X, Y_true in test_dl:
            X, Y_true = to_device(X, Y_true)
            Y_pred = model(X)
            loss = loss_func(Y_pred, Y_true)
            test_loss += loss.item()
            X, Y_true, Y_pred = to_cpu(X, Y_true, Y_pred)
            _, Y_pred = Y_pred.max(dim=1)
            correct += (Y_pred == Y_true).type(torch.IntTensor).sum().item()
            total += len(Y_true)
            accs.append(correct/total)
    return accs


def conv_sec(s):
    "Convert seconds to time format"
    return str(datetime.timedelta(seconds=int(s)))

if verbose: print("     Accuracies per Epoch\n", "-"*32)
train_accs = []
test_accs = []
start_t = time.time()
for epoch in range(epochs):
    # Train
    train_acc = train(model, train_dl, loss_func, opt)
    train_accs += train_acc
    t = time.time() - start_t
    if verbose: print(f"{epoch} | {conv_sec(t)} | train: {train_acc[-1]:.5f}")
    
    # Validate
    test_acc = validate(model, test_dl, loss_func)
    test_accs += test_acc
    t = time.time() - start_t
    if verbose: print(f"{epoch} | {conv_sec(t)} | test:  {test_acc[-1]:.5f}\n", "-"*32)

# Store Accuracies
tr_acc_df = pd.DataFrame(train_accs)
tst_acc_df = pd.DataFrame(test_accs)

if not os.path.isdir('accs'):
    os.mkdir('accs')

tr_acc_df.to_csv(f"accs/train_accs_{opt_t}_{tag}.csv")
tst_acc_df.to_csv(f"accs/test_accs_{opt_t}_{tag}.csv")