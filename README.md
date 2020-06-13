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

(TL;DR – It doesn't).

For the purpose of formality, this is the Nested Lookahead algorithm:

<img src=https://github.com/johngilbert2000/nested_lookahead/blob/master/plots/NLA_Algorithm.png height="312" width="472">

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

**Optimizer Comparison**
- `bash run_cifar.sh` for CIFAR-10 with ResNet-18 (Estimated time: 3 hours with single GPU)
- `bash run_cifar_pullback.sh` for Lookahead pullback/reset momentum on CIFAR-10 (Estimated time: 1 hr. 20 mins.)

Alternatively, use `chmod u+r+x run_cifar.sh; ./run_cifar.sh` instead of `bash run_cifar.sh`.

**Hyperparameter tuning**
- `bash run_cifar_variation.sh`

**Demo**
- `bash run_demo.sh`

Use `make_plot.py` to generate plots in the `plots/` folder. An example usage of `make_plot.py` could be:

```
python make_plot.py --with_t "demo" --tag "demo"
```

Another example, using a specific method:
```
python make_plot.py --method "NestedLookahead Adam" --without "pullback reset" --tag "NLA"
```

Type `python make_plot.py --help` for details on how to use the command line arguments. Alternatively, create plots using the `plot_results.ipynb` Jupyter Notebook.

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

See [report.pdf](https://github.com/johngilbert2000/nested_lookahead/blob/master/report.pdf) for details.

