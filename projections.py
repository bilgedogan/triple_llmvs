import torch
import torch.nn as nn
import torch.nn.functional as F


VISUAL_DIM = 1024
AUDIO_DIM = 512
TEXT_RAW_DIM = 5120
FUSED_DIM = 2048
COMP_DIM = 256


def _kaiming(linear):
    nn.init.kaiming_uniform_(linear.weight, a=0, nonlinearity='linear')
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class TextEncoder(nn.Module):
    """Deterministic txt_t extractor: cat(user_prompt, gen) along token dim,
    channel-pool to 5120, then AdaptiveMaxPool1d to 2048. No learnable params,
    matching the prompt's 'text already 2048d, identity' contract."""

    def __init__(self, out_dim=FUSED_DIM):
        super().__init__()
        self.token_pool = nn.AdaptiveMaxPool1d(1)
        self.dim_pool = nn.AdaptiveMaxPool1d(out_dim)
        self.out_dim = out_dim

    def forward(self, llama_user, llama_gen):
        # inputs: (T, tokens, 5120) each. cat → (T, 2*tokens, 5120)
        x = torch.cat((llama_user, llama_gen), dim=1)
        # channel-pool over tokens: (T, 5120, 2*tokens) → (T, 5120, 1) → (T, 5120)
        x = self.token_pool(x.permute(0, 2, 1)).squeeze(-1)
        # dim-pool 5120 → out_dim
        x = self.dim_pool(x.unsqueeze(1)).squeeze(1)
        return x


class FusionProjections(nn.Module):
    """Visual + audio projections to fused dim (2048). Text path is identity
    (already 2048d via TextEncoder). Outputs are L2-normalised."""

    def __init__(self, visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, out_dim=FUSED_DIM):
        super().__init__()
        self.visual = nn.Linear(visual_dim, out_dim)
        self.audio = nn.Linear(audio_dim, out_dim)
        self.visual_ln = nn.LayerNorm(out_dim)
        self.audio_ln = nn.LayerNorm(out_dim)
        _kaiming(self.visual)
        _kaiming(self.audio)

    def project_visual(self, v):
        x = self.visual_ln(self.visual(v))
        return F.normalize(x, dim=-1)

    def project_audio(self, a):
        x = self.audio_ln(self.audio(a))
        return F.normalize(x, dim=-1)

    def project_text(self, txt):
        # txt already 2048d, L2-normalise to keep modalities on equal footing.
        return F.normalize(txt, dim=-1)

    def forward(self, v, txt, a):
        return self.project_visual(v), self.project_text(txt), self.project_audio(a)


class CompressionProjections(nn.Module):
    """RL state compression: each modality → 256d. Separate params from fusion."""

    def __init__(self, visual_dim=VISUAL_DIM, audio_dim=AUDIO_DIM, text_dim=FUSED_DIM, out_dim=COMP_DIM):
        super().__init__()
        self.visual = nn.Linear(visual_dim, out_dim)
        self.audio = nn.Linear(audio_dim, out_dim)
        self.text = nn.Linear(text_dim, out_dim)
        _kaiming(self.visual)
        _kaiming(self.audio)
        _kaiming(self.text)

    def forward(self, v, txt, a):
        return self.visual(v), self.text(txt), self.audio(a)
