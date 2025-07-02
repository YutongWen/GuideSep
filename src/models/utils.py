import torch
from torch import Tensor
import numpy as np
import os, glob
from tqdm import tqdm

def spec_fwd(spec, spec_abs_exponent=0.5, spec_factor=0.3):
    '''
        # only do this calculation if spec_exponent != 1, 
        # otherwise it's quite a bit of wasted computation
        # and introduced numerical error
    '''
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = spec.abs()**e * torch.exp(1j * spec.angle())
    spec = spec * spec_factor

    return spec
    
def spec_back(spec, spec_abs_exponent=0.5, spec_factor=0.3):
    spec = spec / spec_factor
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())

    return spec

def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

# operations
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
    return gamma

def p_dot_p(t_a, gamma_a, t_b, gamma_b):

    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b , gamma_b, -gamma_a)
    t_max = np.maximum(t_a , t_b)

    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max

    return num / den

def solve_weights(t_i, gamma_i, t_r, gamma_r):

    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)

    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i ))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r ))

    X = np.linalg.solve(A, B)

    return X

def weighted_state_dict(snapshots_list, ema_prof1, ema_prof2, model_weight):

    state_dict_ema = {}
    for i, snapshot in enumerate(tqdm(snapshots_list)):
        ema_prof1.ema_model.load_state_dict(torch.load(snapshot), strict=True)
        ema_prof2.ema_model.load_state_dict(torch.load(snapshot.replace('ema_prof1', 'ema_prof2')), strict=True)

        for key in ema_prof1.ema_model.state_dict().keys():

            ema_1_params = ema_prof1.ema_model.state_dict()[key]
            ema_2_params = ema_prof2.ema_model.state_dict()[key]

            updated_params = model_weight[i][0] * ema_1_params + model_weight[i+len(snapshots_list)][0] * ema_2_params

            if key not in state_dict_ema:
                state_dict_ema[key] = updated_params

            else:
                state_dict_ema[key] += updated_params

    return state_dict_ema


def post_hoc_ema_model(ema_snapshot_path, ema_prof1, ema_prof2, gamma_3):
    # temp, will rewrite
    # compute coefficients for each snapshots and combine
    ema_snapshots = glob.glob(os.path.join(ema_snapshot_path, 'ema_prof1*'))
    ema_snapshots = sorted(ema_snapshots, key=lambda d: int(d.split('_')[-1]))

    ema_snapshots = ema_snapshots[::2]

    
    t_list = [int(snapshot.split('/')[-1].split('_')[-1]) for snapshot in ema_snapshots]

    checkpoint_freq = t_list[1] - t_list[0]
    checkpoint_index = np.arange(t_list[0], t_list[-1] + 1, checkpoint_freq)
    t = np.arange(1, t_list[-1] + 2)

    # by default prof_1 = 16.97 (5%) and prof_2 = 6.94 (10%)
    gamma_1 = 16.97 
    gamma_2 = 6.94
    gammas = np.concatenate(
            (
                np.ones_like(checkpoint_index) * gamma_1,
                np.ones_like(checkpoint_index) * gamma_2,
            )
        )

    t_checkpoint = t[checkpoint_index]
    ts = np.concatenate((t_checkpoint, t_checkpoint))
    last_index = t[-1]
    x = solve_weights(ts, gammas, last_index, gamma_3)

    state_dict_ema = weighted_state_dict(ema_snapshots, ema_prof1, ema_prof2, x)

    return state_dict_ema