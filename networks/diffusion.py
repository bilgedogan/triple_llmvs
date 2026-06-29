import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreDiffusion(nn.Module):
    """Basic DDPM denoiser on the aggregator's scalar per-frame scores.

    Plug-and-play: off unless enabled. Goal = clean noise in importance
    scores. Conditioned on the transformer encoder features. Few noising
    steps (~20-30). At eval it runs SDEdit-style: partially noise the base
    score, then reverse-denoise back to a cleaner score.

    Index/shape convention: x0 is per-frame score (B, T, 1) in [0, 1].
    """

    def __init__(self, feat_dim, num_steps=20, hidden=128, refine_ratio=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.refine_ratio = refine_ratio          # how far to noise base score at eval
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.cond = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU())
        self.t_emb = nn.Embedding(num_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(1 + hidden + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _eps(self, x_t, cond, t):
        # x_t (B,T,1), cond (B,T,H), t (B,T) long -> predicted noise (B,T,1)
        inp = torch.cat([x_t, cond, self.t_emb(t)], dim=-1)
        return self.net(inp)

    def q_sample(self, x0, t, noise):
        ac = self.alphas_cumprod[t].unsqueeze(-1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def loss(self, x, gt):
        """Standard noise-prediction loss. x: encoder feats (B,T,D); gt: (B,T)."""
        cond = self.cond(x)
        x0 = gt.unsqueeze(-1)
        B, T = gt.shape
        t = torch.randint(0, self.num_steps, (B, T), device=gt.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self._eps(x_t, cond, t), noise)

    @torch.no_grad()
    def refine(self, x, s0):
        """SDEdit denoise of base score s0 (B,T) conditioned on feats x (B,T,D)."""
        cond = self.cond(x)
        B, T = s0.shape
        x0 = s0.unsqueeze(-1)
        t_start = max(1, int(round(self.num_steps * self.refine_ratio)))
        t = torch.full((B, T), t_start - 1, device=s0.device, dtype=torch.long)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        for ti in reversed(range(t_start)):
            tt = torch.full((B, T), ti, device=s0.device, dtype=torch.long)
            eps = self._eps(x_t, cond, tt)
            ac = self.alphas_cumprod[tt].unsqueeze(-1)
            x0_pred = ((x_t - (1.0 - ac).sqrt() * eps) / ac.sqrt()).clamp(0.0, 1.0)
            if ti > 0:
                a = self.alphas[tt].unsqueeze(-1)
                beta = self.betas[tt].unsqueeze(-1)
                ac_prev = self.alphas_cumprod[tt - 1].unsqueeze(-1)
                mean = (ac_prev.sqrt() * beta / (1.0 - ac)) * x0_pred \
                    + (a.sqrt() * (1.0 - ac_prev) / (1.0 - ac)) * x_t
                var = beta * (1.0 - ac_prev) / (1.0 - ac)
                x_t = mean + var.sqrt() * torch.randn_like(x_t)
            else:
                x_t = x0_pred
        return x_t.squeeze(-1)
