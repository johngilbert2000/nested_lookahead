# Nested Lookahead

*A Lookahead Optimizer that also uses Lookahead*

Based on the paper by Zhang et al.: [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

Also inspired by Lookahead implementations by:
- Zhang et al.:   https://github.com/michaelrzhang/lookahead
- Liam (alphadl): https://github.com/alphadl/lookahead.pytorch


Goal
------

This project seeks to explore the following question:

If Lookahead improves convergence by seeking out which direction is best for updating weights, would the inner loop of Lookahead benefit by also using Lookahead?


TODO
------
- Implement pullback momentum (pullback, reset)
- Setup and run experiments for Nested Lookahead (CIFAR10, ImageNet, etc.)
- Compare results for differing initial parameters (loop sizes, step sizes, pullback momentum)
- Compare with other optimizers (Regular Lookahead, Adam, SGD)
