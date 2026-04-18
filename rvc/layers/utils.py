from typing import List, Optional, Tuple, Iterator, Union

import torch


def call_weight_data_normal_if_Conv(m: torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        mean = 0.0
        std = 0.01
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation=1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def slice_on_last_dim(
    x: torch.Tensor,
    start_indices: Union[List[int], torch.Tensor],
    segment_size=4,
) -> torch.Tensor:
    new_shape = [*x.shape]
    new_shape[-1] = segment_size
    ret = torch.empty(new_shape, device=x.device)
    for i in range(x.size(0)):
        idx_str = start_indices[i]
        idx_end = idx_str + segment_size
        ret[i, ..., :] = x[i, ..., idx_str:idx_end]
    return ret


def rand_slice_segments_on_last_dim(
    x: torch.Tensor,
    x_lengths: Optional[Union[int, torch.Tensor]] = None,
    segment_size=4,
) -> Tuple[torch.Tensor, Union[List[int], torch.Tensor]]:
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_on_last_dim(x, ids_str, segment_size)
    return ret, ids_str


@torch.jit.script
def activate_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int
) -> torch.Tensor:
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(
    length: torch.Tensor,
    max_length: Optional[int] = None,
):
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def total_grad_norm(
    parameters: Iterator[torch.nn.Parameter],
    norm_type: float = 2.0,
) -> float:
    norm_type = float(norm_type)
    total_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += float(param_norm.item()) ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm
