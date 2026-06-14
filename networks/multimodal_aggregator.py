import torch
import torch.nn as nn
import pytorch_lightning as pl

from projections import FusionProjections, TextEncoder, FUSED_DIM
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary


class MultimodalAggregator(nn.Module):
    """Plain nn.Module aggregator over fused (B,T,2048) sequences.

    Architecture mirrors LLMVS but skips the channel/token pool and the
    5120→2048 projection because the input is already 2048d fused features.
    """

    def __init__(self, reduced_dim=FUSED_DIM, num_heads=2, num_layers=3):
        super().__init__()
        self.reduced_dim = reduced_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim, nhead=num_heads, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim // 2),
            nn.LayerNorm(reduced_dim // 2),
            nn.ReLU(),
            nn.Linear(reduced_dim // 2, reduced_dim // 4),
            nn.ReLU(),
            nn.Linear(reduced_dim // 4, reduced_dim // 8),
            nn.LayerNorm(reduced_dim // 8),
            nn.ReLU(),
            nn.Linear(reduced_dim // 8, reduced_dim // 16),
            nn.ReLU(),
            nn.Linear(reduced_dim // 16, 1),
            nn.Sigmoid(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, fused, mask=None):
        """fused: (B, T, 2048). mask: (B, T) boolean — True for valid frames."""
        key_padding_mask = None
        if mask is not None:
            # nn.TransformerEncoder expects True for *padded* positions.
            key_padding_mask = ~mask
        x = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        scores = self.mlp_head(x).squeeze(-1)
        return scores


def equal_weight_fuse(v_fused, txt, a_fused):
    return (v_fused + txt + a_fused) / 3.0


class PretrainPLModule(pl.LightningModule):
    """Phase 2 wrapper: equal-weight fusion, fusion projections frozen-random,
    train aggregator from scratch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(out_dim=config.reduced_dim)
        self.fusion = FusionProjections(out_dim=config.reduced_dim)
        # in phase 2 pretrain, we now train fusion projections alongside the aggregator:
        # for p in self.fusion.parameters():
        #     p.requires_grad_(False)
        self.aggregator = MultimodalAggregator(
            reduced_dim=config.reduced_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        )
        self.criterion = nn.MSELoss(reduction='none')

    def _build_fused(self, batch):
        v = batch['visual']
        a = batch['audio']
        llama_user = batch['llama_user']
        llama_gen = batch['llama_gen']
        B, T = v.shape[:2]
        # TextEncoder works on (T, tokens, 5120). Flatten batch.
        lu = llama_user.reshape(B * T, llama_user.shape[2], llama_user.shape[3])
        lg = llama_gen.reshape(B * T, llama_gen.shape[2], llama_gen.shape[3])
        txt = self.text_encoder(lu, lg).reshape(B, T, -1)
        if self.config.fusion_mode == 'text_only':
            return txt
        v_fused = self.fusion.project_visual(v)
        a_fused = self.fusion.project_audio(a)
        txt_fused = self.fusion.project_text(txt)
        return equal_weight_fuse(v_fused, txt_fused, a_fused)

    def training_step(self, batch, batch_idx):
        mask = batch['mask'].to(self.device)
        fused = self._build_fused(batch)
        # scores = self.aggregator(fused, mask=mask).clamp(0.0, 1.0)
        scores = self.aggregator(fused, mask=None).clamp(0.0, 1.0)
        gt = batch['gtscore']
#        loss_per = self.criterion(scores, gt)
#        loss = (loss_per * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        loss = self.criterion(scores, gt).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=fused.shape[0])
        return loss

    def _eval_one(self, batch, bidx):
        mask = batch['mask'].to(self.device)
        fused = self._build_fused(batch)
        # scores = self.aggregator(fused, mask=mask).clamp(0.0, 1.0)
        scores = self.aggregator(fused, mask=None).clamp(0.0, 1.0)
        # Only batch size 1 for val/test (variable-length cross-validation).
        score = scores[bidx][mask[bidx]]
        cps = batch['change_points'][bidx]
        n_frames = batch['n_frames'][bidx]
        nfps = batch['n_frame_per_seg'][bidx].tolist()
        picks = batch['picks'][bidx]
        gt_summary = batch['gt_summary'][bidx]
        video_name = batch['video_name'][bidx]
        machine_summary = generate_summary(score, cps, n_frames.unsqueeze(0), nfps, picks)
        kTau, sRho = evaluate_summary(machine_summary, gt_summary, video_name, score, eval_data=self.config.dataset)
        return float(kTau), float(sRho)

    def validation_step(self, batch, batch_idx):
        kTau, sRho = self._eval_one(batch, 0)
        return kTau, sRho

    def validation_epoch_end(self, outs):
        outs = torch.tensor(outs)
        self.log('val_kTau', outs[:, 0].mean(), prog_bar=True)
        self.log('val_sRho', outs[:, 1].mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_one(batch, 0)

    def test_epoch_end(self, outs):
        outs = torch.tensor(outs)
        self.log('val_kTau', outs[:, 0].mean(), prog_bar=True)
        self.log('val_sRho', outs[:, 1].mean(), prog_bar=True)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.config.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
