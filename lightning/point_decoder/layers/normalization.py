import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 3:
            B, N, C = x.shape
            x = x.view(B*N, C)
            x = super().forward(x)
            x = x.view(B, N, C)
        elif x.ndim == 4:
            B, N, K, C = x.shape
            x = x.view(B*N*K, C)
            x = super().forward(x)
            x = x.view(B, N, K, C)
        else:
            raise NotImplementedError
        return x


class CustomGroupNorm1d(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_channels, num_channels)
        
    def forward(self, x):
        if x.ndim == 3:
            B, N, C = x.shape
            x = x.permute(0, 2, 1).contiguous()
            x = super().forward(x)
            x = x.permute(0, 2, 1).contiguous()
        elif x.ndim == 4:
            B, N, K, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            x = super().forward(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise NotImplementedError
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input, offsets):
        with torch.cuda.amp.autocast(enabled=False):
            offsets = torch.cat([offsets.new_zeros(1), offsets]).long()
            
            # Compute the mean and standard deviation per segment
            mean = torch_scatter.segment_csr(input, offsets, reduce='mean')
            mean_gathered = torch_scatter.gather_csr(mean, offsets)
            
            variance = torch_scatter.segment_csr((input - mean_gathered) ** 2, offsets, reduce='mean')
            std = torch.sqrt(variance + self.eps)
            std_gathered = torch_scatter.gather_csr(std, offsets)
            
            # Normalize the input
            output = (input - mean_gathered) / std_gathered
            
            if self.elementwise_affine:
                output = output * self.weight + self.bias
        
        return output
    

class AdaLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, w_shape, eps=1e-5):
        super().__init__(
            normalized_shape, 
            eps, 
            elementwise_affine=False
        )

        self.affine = nn.Linear(w_shape, normalized_shape)
    
    def forward(self, feat, global_feat, offset):
        global_feat = torch_scatter.gather_csr(
            src=self.affine(global_feat),
            indptr=F.pad(offset, (1, 0), "constant", 0)
        )
        return global_feat * super().forward(feat)
