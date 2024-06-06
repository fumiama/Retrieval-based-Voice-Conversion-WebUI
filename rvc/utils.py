from typing import List, Optional, Tuple

import torch

def call_weight_data_normal_if_Conv(m: torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        mean=0.0
        std=0.01
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def slice_on_last_dim(
        x: torch.Tensor, start_indices: List[int], segment_size=4,
    ) -> torch.Tensor:
    new_shape = x.shape
    new_shape[-1] = segment_size
    ret = torch.empty(new_shape)
    for i in range(x.size(0)):
        idx_str = start_indices[i]
        idx_end = idx_str + segment_size
        ret[i, ..., :] = x[i, ..., idx_str:idx_end]
    return ret


def rand_slice_segments(
        x: torch.Tensor, x_lengths: int = None, segment_size=4,
    ) -> Tuple[torch.Tensor, List[int]]:
    b, _, t = x.size()
    if x_lengths is None: x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_on_last_dim(x, ids_str, segment_size)
    return ret, ids_str


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    return torch.tensor(pad_shape).flip(0).reshape(-1).int().tolist()


def sequence_mask(
        length: torch.Tensor, max_length: Optional[int] = None,
    ) -> torch.BoolTensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
