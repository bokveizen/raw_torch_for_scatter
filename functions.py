# scatter_add

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import torch_scatter


# torch_scatter/utils.py
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


# torch_scatter/utils.py
def scatter_sum_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


# torch_scatter/utils.py
def scatter_add_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_sum_raw(src, index, dim, out, dim_size)


def scatter_mean_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum_raw(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum_raw(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


# value only, equivalent to scatter_max(...)[0]
def scatter_max_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_reduce(0, index=index, src=src, reduce="amax", include_self=False)
        
    else:
        return out.scatter_reduce(
            0, index=index, src=src, reduce="amax", include_self=False
        )


# value only, equivalent to scatter_min(...)[0]
def scatter_min_raw(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    pass
    

def scatter_mul_raw(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    pass


def scatter_softmax_raw(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    pass


def scatter_log_softmax_raw(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                        eps: float = 1e-12,
                        dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_log_softmax` can only be computed over '
                         'tensors with floating point data types.')
    pass


def scatter_logsumexp_raw(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                      out: Optional[torch.Tensor] = None,
                      dim_size: Optional[int] = None,
                      eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    pass


def scatter_std_raw(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                unbiased: bool = True) -> torch.Tensor:

    pass