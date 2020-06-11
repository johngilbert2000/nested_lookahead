# Nested Lookahead

*A Lookahead Optimizer that also uses Lookahead*

Based on the paper by Zhang et al.: [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

Also inspired by Lookahead implementations by:
- Zhang et al.:   https://github.com/michaelrzhang/lookahead
- Liam (alphadl): https://github.com/alphadl/lookahead.pytorch


Goal
------

This project seeks to explore the following question:

> If Lookahead improves convergence by seeking out which direction is best for updating weights, would the inner loop of Lookahead benefit by also using Lookahead?

TL;DR – It doesn't.

Usage
------
To use `NestedLookahead`, set it up as shown, then use it as a regular Pytorch optimizer (`optimizer.zero_grad()`, `optimizer.step()`, etc.)
```python
import torch
from NestedLookahead import NestedLookahead

inner_optimizer = torch.optim.Adam(your_model.parameters()) # <-- choose any Pytorch optimizer
optimizer = NestedLookahead(opt=inner_optimizer, lr=0.001, k=5, s=5, pullback='pullback')
```

The following versions were used in development:
```
Python 3.7.7
Cuda 10.0
Nvidia Driver: 410.78

torch 1.1.0
torchvision 0.3.0
pandas 1.0.3
numpy 1.18.1
matplotlib 3.1.3
```

The experiments shown below were run on a GeForce RTX 2080 Ti GPU.

Experiments
------
From the command line, run the following shell scripts to run full experiments with Nested Lookahead, Lookahead, SGD, and Adam.

- `bash run_cifar.sh` for CIFAR-10 with ResNet-18 (Estimated time: 3 hours with single GPU)
- `bash run_cifar_pullback.sh` for Lookahead pullback/reset momentum on CIFAR-10 (Estimated time: 1 hr. 20 mins.)

Alternatively, use `chmod u+r+x run_cifar.sh; ./run_cifar.sh` instead of `bash run_cifar.sh`.

Parameters
------

Feel free to change the following parameters in the bash scripts:
```python
epochs=20 # number of epochs
bs=64 # batch size
lr=0.001 # learning rate
momentum=0.9 # momentum (for SGD)
weight_decay=0.001 # weight decay

# Lookahead Only
k=5 # number of innermost fast-weight steps
a=0.5 # alpha; (inner) slow-weight step size
s=5 # number of outer slow weight steps (Nested Lookahead only)
h=0.5 # outer slow weight step size (Nested Lookahead only)
pullback="None" # Lookahead pullback momentum ("None", "pullback", or "reset")
```

Results
------

The following are results for all optimizers on CIFAR-10 with ResNet-18:
```
folds:         3
epochs:        20
batch size:    64
learning rate: 0.001
momentum:      0.9 (for SGD)
weight decay:  0.001

k: 5   (fast-weight steps)
a: 0.5 (inner slow-weight step size)
s: 5   (outer slow weight steps (Nested Lookahead only))
h: 0.5 (outer slow weight step size (Nested Lookahead only))
pullback: "None", "reset", and "pullback" (Lookahead pullback momentum)
```
![Cifar10_test_accuracies](https://github.com/johngilbert2000/nested_lookahead/blob/master/plots/cifar10_default_test_acc.png)

![Cifar10_test_acc_tail](https://github.com/johngilbert2000/nested_lookahead/blob/master/plots/cifar10_default_test_acc_tail.png)

Conclusion
------
`NestedLookahead` appears to have no performance improvement over regular `Lookahead`.

TODO
------
- Setup and run other experiments for Nested Lookahead (CIFAR-100, ImageNet, etc.)
- Compare results for differing initial parameters (loop sizes, step sizes, pullback momentum)
