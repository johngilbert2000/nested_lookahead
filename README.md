# Nested Lookahead

*A Lookahead Optimizer that also uses Lookahead*

<hr>

Based on the paper by Zhang et al.: [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

Also inspired by Lookahead implementations by:
- Zhang et al.:   https://github.com/michaelrzhang/lookahead
- Liam (alphadl): https://github.com/alphadl/lookahead.pytorch


This project seeks to explore the following question:

If Lookahead improves convergence by seeking out which direction is best for updating weights, would the inner loop of Lookahead benefit by also using Lookahead?
