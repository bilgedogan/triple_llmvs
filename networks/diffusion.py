"""Plug-and-play diffusion denoiser for transformer encoder outputs.

Drop-in module that refines per-frame feature sequences produced by the
aggregation transformer before they reach the scoring head. Trained as an
auxiliary DDPM denoiser; at inference it applies SDEdit-style light denoising
(noise to a partial timestep, then reverse) to remove noise in the encoder
outputs. Gated by `config.use_diffusion` so it can be added or removed without
touching the base model behaviour.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t: [B] long/float -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DenoiseNet(nn.Module):
    """Predict the noise eps added to a feature sequence at timestep t.

    Whole [F, D] feature map is one diffusion sample at a single timestep t.
    A small transformer mixes information across frames for context.
    """

    def __init__(self, dim, num_heads=2, num_layers=2, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.in_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True
        )
        self.net = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(dim, dim)

    def forward(self, x_t, t):
        # x_t: [F, D], t: [1] -> eps: [F, D]
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))  # [1, D]
        h = self.in_norm(x_t) + temb  # broadcast t over frames
        h = self.net(h.unsqueeze(0)).squeeze(0)
        return self.out(h)


class FeatureDiffusion(nn.Module):
    """DDPM over encoder feature sequences with SDEdit refinement at inference.

    Usage (plug-and-play):
        diff = FeatureDiffusion(dim=D)
        loss = diff.loss(h)          # aux denoise loss during training
        h_clean = diff.refine(h)     # remove noise at eval
    """

    def __init__(self, dim, timesteps=1000, num_heads=2, num_layers=2,
                 refine_strength=0.3):
        super().__init__()
        self.timesteps = timesteps
        self.refine_strength = refine_strength
        self.net = DenoiseNet(dim, num_heads, num_layers)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('acp', acp)
        self.register_buffer('sqrt_acp', torch.sqrt(acp))
        self.register_buffer('sqrt_om_acp', torch.sqrt(1.0 - acp))

    def q_sample(self, x0, t, noise):
        """Forward diffuse x0 to timestep t (scalar index)."""
        return self.sqrt_acp[t] * x0 + self.sqrt_om_acp[t] * noise

    def loss(self, x0):
        """Auxiliary noise-prediction loss for a single [F, D] feature map."""
        t = torch.randint(0, self.timesteps, (1,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.net(x_t, t)
        return F.mse_loss(pred, noise)

    def refine(self, x0):
        """SDEdit denoise: noise x0 to a partial timestep, then reverse to 0.

        refine_strength in (0, 1] sets how far to noise before denoising;
        small values keep refined ~ x0 (light denoise), avoiding train/eval
        distribution shift.
        """
        t0 = max(1, int(self.refine_strength * self.timesteps))
        x = self.q_sample(
            x0, torch.tensor(t0 - 1, device=x0.device), torch.randn_like(x0)
        )
        for i in reversed(range(t0)):
            t = torch.tensor([i], device=x0.device)
            eps = self.net(x, t)
            beta, alpha = self.betas[i], self.alphas[i]
            mean = (x - (beta / self.sqrt_om_acp[i]) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x
