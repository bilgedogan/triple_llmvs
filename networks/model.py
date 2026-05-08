import torch
from torch import nn

import pytorch_lightning as pl
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary
from pytorch_lightning import seed_everything
import numpy as np
import torch.nn.functional as F
from triplesumm.model import TripleSumm

seed_everything(1112)

class LLMVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.d_linear1 = nn.Linear(5120, self.config.reduced_dim)
        self.d_linear1_norm = nn.LayerNorm(self.config.reduced_dim)

        self.triplesumm = TripleSumm(
            visual_dim=getattr(config, 'visual_dim', 1042),
            text_dim=self.config.reduced_dim,
            audio_dim=getattr(config, 'audio_dim', 512),
            input_dim=getattr(config, 'input_dim', 128),
            hidden_dim=getattr(config, 'hidden_dim', 192),
            num_model_layers=getattr(config, 'num_layers', 2),
            num_mst_layers=getattr(config, 'num_mst_layers', 2),
            num_cmf_layers=getattr(config, 'num_cmf_layers', 2),
            num_heads=getattr(config, 'num_heads', 4),
            dropout=getattr(config, 'dropout', 0.3),
            window_size=getattr(config, 'window_size', [5, 15, 45, 0]),
            max_seq_len=getattr(config, 'max_seq_len', 1000),
            get_attn_weights=False,
        )

        self.criterion = nn.MSELoss()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, visual, audio, mask=None):
        # x: [T, 2*tokens, 5120]
        x = self.c_max_pooling(x.permute(0, 2, 1)).squeeze(2)  # [T, 5120]
        x = self.d_linear1(x)           # [T, reduced_dim]
        x = self.d_linear1_norm(x)      # [T, reduced_dim]

        text = x.unsqueeze(0)           # [1, T, reduced_dim]
        visual = visual.unsqueeze(0)    # [1, T, 1042]
        audio = audio.unsqueeze(0)      # [1, T, 512]

        out, _ = self.triplesumm(visual, text, audio, mask)
        return out  # [1, T]

    @staticmethod
    def _to_np(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return np.array(x)

    @staticmethod
    def _compute_metrics(score_np, machine_summary, gt_np, picks, video_name, dataset):
        machine_summary = LLMVS._to_np(machine_summary)
        gt_np           = LLMVS._to_np(gt_np)
        score_np        = LLMVS._to_np(score_np)

        kTau, sRho = evaluate_summary(machine_summary, gt_np, video_name, score_np, eval_data=dataset)

        # Flatten multi-annotator gt → 1D mean
        gt_1d = gt_np.mean(axis=0) if gt_np.ndim > 1 else gt_np

        # F1 and mp15 from knapsack-selected machine_summary vs gt
        ms = machine_summary.astype(bool)
        gt_bin = (gt_1d > 0.5).astype(bool)
        min_len = min(len(ms), len(gt_bin))
        ms, gt_bin = ms[:min_len], gt_bin[:min_len]

        tp = float((ms & gt_bin).sum())
        fp = float((ms & ~gt_bin).sum())
        fn = float((~ms & gt_bin).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        mp15 = prec  # precision at 15% knapsack budget

        # map50: frame-level AP from score ranking vs subsampled gt
        gt_full = gt_1d.astype(float)
        picks_arr = picks.cpu().numpy() if hasattr(picks, 'cpu') else np.array(picks)
        gt_sub = gt_full[picks_arr]

        sorted_idx = np.argsort(-score_np)
        gt_sorted = gt_sub[sorted_idx]
        cumtp = np.cumsum(gt_sorted)
        prec_at_k = cumtp / (np.arange(len(gt_sorted)) + 1)
        map50 = float(np.sum(prec_at_k * gt_sorted) / (gt_sub.sum() + 1e-8))

        return f1, float(kTau), float(sRho), mp15, map50

    def training_step(self, train_batch, batch_idx):
        x1 = train_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = train_batch['llama_embedding_generation'].squeeze(0)
        x = torch.cat((x1, x2), dim=1)

        visual = train_batch['visual_feat'].squeeze(0)
        audio  = train_batch['audio_feat'].squeeze(0)
        y    = train_batch['gtscore']
        mask = train_batch['mask']

        score = self.forward(x, visual, audio, mask=mask).clamp(0.0, 1.0)
        loss = self.criterion(score, y).mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=1)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)
        x = torch.cat((x1, x2), dim=1)

        visual = val_batch['visual_feat'].squeeze(0)
        audio  = val_batch['audio_feat'].squeeze(0)
        mask   = val_batch['mask']

        score = self.forward(x, visual, audio, mask=mask).clamp(0.0, 1.0).squeeze()

        gt_summary = val_batch['gt_summary'][0]
        cps        = val_batch['change_points'][0]
        n_frames   = val_batch['n_frames']
        nfps       = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]
        picks      = val_batch['picks'][0]

        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
        score_np = score.cpu().numpy()
        gt_np    = gt_summary.cpu().numpy() if hasattr(gt_summary, 'cpu') else np.array(gt_summary)

        return self._compute_metrics(score_np, machine_summary, gt_np, picks, video_name, self.config.dataset)

    def validation_epoch_end(self, outs):
        outs  = torch.tensor(outs)
        f1    = outs[:, 0].mean()
        kTau  = outs[:, 1].mean()
        sRho  = outs[:, 2].mean()
        mp15  = outs[:, 3].mean()
        map50 = outs[:, 4].mean()

        self.log('val_f1',    f1,    on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_kTau',  kTau,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho',  sRho,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mp15',  mp15,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_map50', map50, on_step=False, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()

    def test_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)
        x = torch.cat((x1, x2), dim=1)

        visual = val_batch['visual_feat'].squeeze(0)
        audio  = val_batch['audio_feat'].squeeze(0)
        mask   = val_batch['mask']

        score = self.forward(x, visual, audio, mask=mask).clamp(0.0, 1.0).squeeze()

        gt_summary = val_batch['gt_summary'][0]
        cps        = val_batch['change_points'][0]
        n_frames   = val_batch['n_frames']
        nfps       = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]
        picks      = val_batch['picks'][0]

        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
        score_np = score.cpu().numpy()
        gt_np    = gt_summary.cpu().numpy() if hasattr(gt_summary, 'cpu') else np.array(gt_summary)

        return self._compute_metrics(score_np, machine_summary, gt_np, picks, video_name, self.config.dataset)

    def test_epoch_end(self, outs):
        outs  = torch.tensor(outs)
        f1    = outs[:, 0].mean()
        kTau  = outs[:, 1].mean()
        sRho  = outs[:, 2].mean()
        mp15  = outs[:, 3].mean()
        map50 = outs[:, 4].mean()

        self.log('val_f1',    f1,    on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_kTau',  kTau,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho',  sRho,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mp15',  mp15,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_map50', map50, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
