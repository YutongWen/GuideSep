import torch
import torch.nn as nn
from .operator_utils import lengths_to_mask

class MaskedGroupNorm2d(nn.GroupNorm):
    """
    Masked verstion of the Group normalization.
    https://github.com/pytorch/audio/issues/2242
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's GroupNorm implementation for argument details.
    """
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(MaskedGroupNorm2d, self).__init__(
            num_groups,
            num_features,
            eps,
            affine
        )

    def forward(self, inp, lengths):
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        
        assert inp.shape[1] % self.num_groups == 0, 'Feature size not divisible by groups'

        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        
        ave_mask = mask / lengths[:, None] / (inp.shape[1] / self.num_groups) / inp.shape[-2] #also features
        ave_mask = ave_mask.unsqueeze(1).unsqueeze(2)

        # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
        # variance, we do not need to make any tensor shape manipulation.
        # mean = E[X] is simply the sum-product of our "probability" mask with the input...
         #mask out any extra bits of data - such as those left from conv bleeding

        inp_r = inp.reshape([inp.shape[0], self.num_groups, -1, inp.shape[-2], inp.shape[-1]])
        ave_mask = ave_mask.unsqueeze(1)

        mean = (ave_mask * inp_r).sum([2, 3, 4])
        # ...whereas Var(X) is directly derived from the above formulae
        # This should be numerically equivalent to the biased sample variance
        var = (ave_mask * (inp_r ** 2)).sum([2, 3, 4]) - mean ** 2
        
        inp_r = (inp_r - mean[:, :, None, None, None]) / (torch.sqrt(var[:, :, None, None, None] + self.eps))
        out = inp_r.reshape(inp.shape)
        
        if self.affine:
            out = out * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return out * mask.unsqueeze(1).unsqueeze(1)
        
class MaskedGroupNorm1d(nn.GroupNorm):
    """
    Masked verstion of the Group normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's GroupNorm implementation for argument details.
    """
    def __init__(self, num_groups, num_features, eps=1e-5,affine=True):
        super(MaskedGroupNorm1d, self).__init__(
            num_groups,
            num_features,
            eps,
            affine
        )

    def forward(self, inp, lengths):
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        
        assert inp.shape[1] % self.num_groups == 0, 'Feature size not divisible by groups'

        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        ave_mask = mask / lengths[:,None] / (inp.shape[-2] / self.num_groups) #also features
        ave_mask = ave_mask.unsqueeze(1)#.expand(inp.shape)

        # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
        # variance, we do not need to make any tensor shape manipulation.
        # mean = E[X] is simply the sum-product of our "probability" mask with the input...
        inp = inp*mask.unsqueeze(1) #mask out any extra bits of data - such as those left from conv bleeding
        inp_r = inp.reshape([inp.shape[0],self.num_groups,-1,inp.shape[-1]])
        ave_mask = ave_mask.unsqueeze(2)
        mean = (ave_mask * inp_r).sum([2, 3])
        # ...whereas Var(X) is directly derived from the above formulae
        # This should be numerically equivalent to the biased sample variance
        var = (ave_mask * inp_r ** 2).sum([2, 3]) - mean ** 2

        inp_r = (inp_r - mean[:,:,None,None]) / (torch.sqrt(var[:, :, None, None] + self.eps))
        out = inp_r.reshape(inp.shape)

        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * mask.unsqueeze(1)
    

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
    