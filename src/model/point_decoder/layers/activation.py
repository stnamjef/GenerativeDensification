import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 


class _TruncExp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


class TruncExp(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift
    
    def forward(self, x):
        return _TruncExp.apply(x - self.shift)


class Normalize(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)