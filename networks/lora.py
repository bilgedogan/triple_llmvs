import math

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class LoRADelta(nn.Module):
    """Additive low-rank delta: W' = W + (alpha/rank) * (B @ A).

    Used via torch.nn.utils.parametrize to keep base weight frozen while only
    A,B receive gradients.
    """

    def __init__(self, weight_shape, rank=8, alpha=16):
        super().__init__()
        out_f, in_f = weight_shape
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero so delta starts at 0.
        self.scale = alpha / rank

    def forward(self, W):
        return W + self.scale * (self.B @ self.A)


def apply_lora_to_aggregator(aggregator, rank=8, alpha=16):
    """Attach LoRA deltas to every self-attention in_proj + out_proj.

    Returns the list of LoRA parameter modules so callers can build an
    optimizer over only LoRA params.
    """
    # Freeze every base parameter first; we'll selectively unfreeze the LoRA
    # A/B tensors after registration.
    for p in aggregator.parameters():
        p.requires_grad_(False)

    lora_params = []
    for layer in aggregator.transformer.layers:
        attn = layer.self_attn
        ip = LoRADelta(attn.in_proj_weight.shape, rank=rank, alpha=alpha)
        ip = ip.to(attn.in_proj_weight.device, dtype=attn.in_proj_weight.dtype)
        parametrize.register_parametrization(attn, 'in_proj_weight', ip)
        lora_params.append(ip)
        op = LoRADelta(attn.out_proj.weight.shape, rank=rank, alpha=alpha)
        op = op.to(attn.out_proj.weight.device, dtype=attn.out_proj.weight.dtype)
        parametrize.register_parametrization(attn.out_proj, 'weight', op)
        lora_params.append(op)

    # Unfreeze the LoRA A/B params only.
    for m in lora_params:
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)
    return lora_params


def lora_parameters(lora_modules):
    for m in lora_modules:
        yield m.A
        yield m.B
