import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from NestedLookahead import NestedLookahead

import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import argparse

import gc
gc.collect()

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
#parser.add_argument("--verbose", help="prints loss and accuracy", type=bool, default=True)
args = parser.parse_args()

# Parameters
#epochs = 5 # 30
#bs = 64 # 128
#lr = 0.001
#momentum = 0.9
#weight_decay = 1e-3
#opt_t = 'Adam' # optimizer type

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

k = args.k
a = args.a
s = args.s
h = args.h
pullback = None if 'None' in args.pullback else args.pullback

if (opt_t == 'Lookahead_SGD') or (opt_t == 'Lookahead_Adam'):
    s = 0

loss_func = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_path = './cifar_data'
verbose = True

def to_device(*args):
    return [obj.to(device) for obj in args]

# Transformations performed on data
trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0, contrast=(0.7,1.), saturation=(0.7,1.), hue=(0.,0.1)),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and std taken from imagenet defaults
#     transforms.RandomErasing(p=0.5, scale=(0.02,0.10), value=128)
        ])

train_ds = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=trans)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)

test_ds = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=trans)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)

# Model
model = models.resnet18(pretrained=False)
model = model.to(device)

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

# Training
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
        _, Y_pred = Y_pred.max(1)
        correct += (Y_pred == Y_true).type(torch.DoubleTensor).sum().item()
        total += len(Y_true)
        accs.append(correct/total)
    return accs

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
            _, Y_pred = Y_pred.max(1)
            correct += (Y_pred == Y_true).type(torch.DoubleTensor).sum().item()
            total += len(Y_true)
            accs.append(correct/total)
    return accs

train_accs = []
test_accs = []
start_t = time.time()
for epoch in range(epochs):
    # train loop
    train_acc = train(model, train_dl, loss_func, opt)
    train_accs += train_acc
    t = time.time() - start_t
    if verbose: print(epoch, int(t), "s , train_acc:", train_acc[-1])
    
    # validation loop
    test_acc = validate(model, test_dl, loss_func)
    test_accs += test_acc
    t = time.time() - start_t
    if verbose: print(epoch, int(t), "s, test_acc:", test_acc[-1])

tr_acc_df = pd.DataFrame(train_accs)
tst_acc_df = pd.DataFrame(test_accs)

if not os.path.isdir('accs'):
    os.mkdir('accs')

tr_acc_df.to_csv(f"accs/train_accs_{opt_t}_{tag}.csv")
tst_acc_df.to_csv(f"accs/test_accs_{opt_t}_{tag}.csv")