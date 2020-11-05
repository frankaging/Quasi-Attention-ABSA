import torch
import torch.nn as nn
import numpy as np
from numpy import newaxis as na


##############################################################################
#
# The function to back-out layerwise attended relevance scores.
#
##############################################################################

def l_lap_grad(post_hs, pre_hs, post_A, eps=1e-7, bias_factor=1.0, debug=False):
    '''
    sometimes vectorize just requires more powerful memory, we can use loop
    e.g., (b, seq_l, d_hid)
    '''

    alpha = bias_factor
    beta = 1 - alpha
    crop_function = abs

    pre_hs_positive = pre_hs + eps
    post_hs_err = post_hs + eps

    s_positive = alpha * post_A / post_hs_err
    positive_relevances = torch.autograd.grad(post_hs, pre_hs, grad_outputs=s_positive)[0]
    inp_relevances = pre_hs_positive * positive_relevances

    # rescale
    inp_relevances = crop_function(inp_relevances)
    ref_scale = torch.sum(post_A, dim=-1, keepdim=True)
    inp_scale = torch.sum(inp_relevances, dim=-1, keepdim=True) + eps
    scaler = ref_scale / inp_scale

    inp_relevances = inp_relevances * scaler

    return inp_relevances

def a_lap_vectorize(post_hs, pre_hs, attn_hs, post_A, eps=1e-6, bias=0.0, bias_factor=1.0, debug=False):
    '''
    to reduce the runtime, we vectorize it to run it faster
    assuming the input tensor is a 4d tensor with
    e.g., (b, n_head, seq_l, d_hid)
    if n_head should be 1 incase of local attention tracing
    '''
    # simple workaround for determining the device
    cuda_check = post_hs.is_cuda
    pos_unit = torch.tensor(1.)
    neg_unit = torch.tensor(-1.)
    if cuda_check:
        get_cuda_device = post_hs.get_device()
        pos_unit = pos_unit.to(get_cuda_device)
        neg_unit = neg_unit.to(get_cuda_device)

    seq_l = post_hs.shape[2]
    attn_hs_T = attn_hs.transpose(2,3).contiguous()
    attn_hs_T_expand = attn_hs_T.unsqueeze(dim=-1)
    pre_hs_expand = pre_hs.unsqueeze(dim=3)
    numer_vec = pre_hs_expand * attn_hs_T_expand
    # stablizing
    sign_out = torch.where(post_hs>=0, pos_unit, neg_unit)
    post_hs += eps * sign_out
    post_hs_expand = torch.stack(seq_l*[post_hs], dim=2)
    sign_out_expand = torch.stack(seq_l*[sign_out], dim=2)
    numer_vec += ( eps * sign_out_expand / (seq_l * post_hs.shape[-1]) )
    message_vec = numer_vec / post_hs_expand
    post_A_expand = torch.stack(seq_l*[post_A], dim=2)
    pre_A = message_vec * post_A_expand
    pre_A = pre_A.sum(dim=3)
    return pre_A