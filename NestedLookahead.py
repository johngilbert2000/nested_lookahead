from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer

#   Lookahead paper by Zhang et al.:   https://arxiv.org/abs/1907.08610

# Based on work by:
#   Zhang et al.:   https://github.com/michaelrzhang/lookahead (code)
#   Liam (alphadl): https://github.com/alphadl/lookahead.pytorch (code)


class NestedLookahead(torch.optim.Optimizer):
    """
    A Nested Lookahead optimizer, in which the lookahead loop uses lookahead

    opt: optimizer for fast weights
    k: number of fast steps before updating inner slow step
    s: number of inner (nested) slow steps before updating outer slow step
    a: inner slow weight step size, alpha ∈ [0,1]
    b: outer slow weight step size, beta ∈ [0,1]
    pullback: pullback momentum, ("reset", "pullback", or None)

    Note: Becomes original lookahead optimizer if s == 0
    """
    def __init__(self, opt, k=5, s=0, a=0.3, b=0.7, pullback=None):
        if isinstance(pullback, str):
            pullback = pullback.lower()
        
        if (k < 1):
            raise ValueError("Fast k steps must be a natural number")
        elif (s < 0):
            raise ValueError("Inner slow s steps must be a natural number")
        elif (a < 0) or (a > 1):
            raise ValueError("Requirement: a ∈ [0,1]")
        elif (b < 0) or (b > 1):
            raise ValueError("Requirement: b ∈ [0,1]")
        elif (pullback not in ["reset", "pullback", None]):
            raise ValueError("Pullback momentum options: 'reset', 'pullback', or None")
        
        self.opt = opt
        self.s = int(s)
        self.k = int(k)
        self.a = a
        self.b = b
        self.pullback = pullback
        self._k_ct = 0 # counter for k
        self._s_ct = 0 # counter for s
        
        self.param_groups = self.opt.param_groups
        self.state = defaultdict(dict)
        
    def __getstate__(self):
        return {
            'state': self.state,
            'opt': self.opt,
            'a': self.a,
            'b': self.b,
            'pullback': self.pullback,
            'k': self.k,
            's': self.s,
            '_k_ct': self._k_ct,
            '_s_ct': self._s_ct,
        }

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)
        
    def add_param_group(self, param_group):
        self.opt.add_param_group(param_group)
        
    def zero_grad(self):
        self.opt.zero_grad()
        
    def update_inner(self, group):
        "Update inner slow weights with fast weights using lookahead"
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state.keys():
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += self.a * (fast.data - slow)
            fast.data.copy_(slow)
        
    def update_outer(self, group):
        "Update outer slow weights with inner slow weights using lookahead"
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state.keys():
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            
            slow = param_state["slow_param"]
            if "outer_param" not in param_state.keys():
                param_state["outer_param"] = torch.zeros_like(slow)
                param_state["outer_param"].copy_(slow)
            outer = param_state["outer_param"]
            outer += self.b * (slow - outer)
            fast.data.copy_(outer)
    
    def update(self):
        for group in self.param_groups:
            if self._k_ct == 0:
                self.update_inner(group) # nested lookahead update
                self._s_ct += 1
            if (self._s_ct == 0) and (self.s > 0):
                self.update_outer(group) # outer lookahead update
            self._k_ct += 1
            
            # reset counter at end of loop
            if self._s_ct >= self.s:
                self._s_ct = 0
            if self._k_ct >= self.k:
                self._k_ct = 0

    def step(self, closure=None):
        loss = self.opt.step(closure)
        self.update()
        return loss