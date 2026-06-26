import torch
from torch import nn
from einops import repeat

import pytorch_lightning as pl
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary
from networks.diffusion import FeatureDiffusion
from pytorch_lightning import seed_everything
import pdb
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
seed_everything(1112)

class LLMVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.d_max_pooling = nn.AdaptiveMaxPool1d(self.config.reduced_dim)
        self.d_linear1 = nn.Linear(5120, self.config.reduced_dim)
        self.d_linear1_norm = nn.LayerNorm(self.config.reduced_dim)
        self.c_max_pooling = nn.AdaptiveMaxPool1d(1)
            
        encoder_layer_agg = nn.TransformerEncoderLayer(d_model = self.config.reduced_dim, nhead = self.config.num_heads, batch_first = True)
        self.transformer_encoder_agg = nn.TransformerEncoder(encoder_layer_agg, num_layers=self.config.num_layers)

        self.mlp_head = torch.nn.Sequential(
            nn.Linear(self.config.reduced_dim, self.config.reduced_dim//2),
            nn.LayerNorm(self.config.reduced_dim // 2), 
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//2, self.config.reduced_dim//4),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//4, self.config.reduced_dim//8),
            nn.LayerNorm(self.config.reduced_dim // 8),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//8, self.config.reduced_dim//16),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//16, 1),
            nn.Sigmoid()
        )
                
        self.criterion = nn.MSELoss()

        # Plug-and-play diffusion denoiser on encoder outputs (default off).
        self.use_diffusion = getattr(self.config, 'use_diffusion', False)
        self.diff_loss_weight = getattr(self.config, 'diff_loss_weight', 1.0)
        if self.use_diffusion:
            self.diffusion = FeatureDiffusion(
                dim=self.config.reduced_dim,
                timesteps=getattr(self.config, 'diff_timesteps', 1000),
                num_heads=self.config.num_heads,
                num_layers=getattr(self.config, 'diff_layers', 2),
                refine_strength=getattr(self.config, 'diff_strength', 0.3),
            )

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        """Map raw concatenated embeddings to per-frame encoder features [F, D]."""
        x = self.c_max_pooling(x.permute(0,2,1)).squeeze(2)

        x = self.d_linear1(x)
        x = self.d_linear1_norm(x)
        x = x.unsqueeze(0)

        x = self.transformer_encoder_agg(x).squeeze(0)
        return x

    def forward(self, x, mask):
        h = self.encode(x)
        # Refine encoder outputs with diffusion at inference (train uses raw h).
        if self.use_diffusion and not self.training:
            h = self.diffusion.refine(h)
        return self.mlp_head(h)
    
    def training_step(self, train_batch, batch_idx):
        x1 = train_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = train_batch['llama_embedding_generation'].squeeze(0)
        
        x = torch.cat((x1, x2), dim=1)  

        y = train_batch['gtscore']
        mask = train_batch['mask']

        h = self.encode(x)
        score = self.mlp_head(h).squeeze(1).unsqueeze(0)
        del x, mask
        score = score.clamp(0.0, 1.0)

        loss = self.criterion(score, y).mean()

        if self.use_diffusion:
            # Aux denoise loss; detach so diffusion stays a pluggable add-on
            # and does not reshape the base encoder/scoring objective.
            diff_loss = self.diffusion.loss(h.detach())
            loss = loss + self.diff_loss_weight * diff_loss
            self.log('diff_loss', diff_loss, on_step=True, on_epoch=True, batch_size = 1)

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size = 1)

        torch.cuda.empty_cache()
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)
        
        x = torch.cat((x1, x2), dim=1)  
        mask = val_batch['mask']

        score = self.forward(x, mask=mask).squeeze(1).unsqueeze(0)
        del x, mask
        score = score.clamp(0.0, 1.0)

        score = score.squeeze()
        gt_summary = val_batch['gt_summary'][0]
        cps = val_batch['change_points'][0]
        n_frames = val_batch['n_frames']
        nfps = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]
 
        picks = val_batch['picks'][0]
        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)

        kTau, sRho = evaluate_summary(machine_summary, gt_summary, video_name, score, eval_data=self.config.dataset)

        return kTau, sRho

    def validation_epoch_end(self, outs):
        outs = torch.tensor(outs)
        
        kTau = outs[:,0].mean()
        sRho = outs[:,1].mean()

        self.log('val_kTau', kTau, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho', sRho, on_step=False, on_epoch=True, prog_bar=True)

        torch.cuda.empty_cache()

    def test_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)
        
        x = torch.cat((x1, x2), dim=1)  
        y = val_batch['gtscore'].squeeze(0)
        mask = val_batch['mask']

        score = self.forward(x, mask=mask).squeeze(1).unsqueeze(0)
        del x, mask
        score = score.clamp(0.0, 1.0)

        score = score.squeeze()

        gt_summary = val_batch['gt_summary'][0]
        cps = val_batch['change_points'][0]
        n_frames = val_batch['n_frames']
        nfps = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]

        picks = val_batch['picks'][0]
        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)

        kTau, sRho = evaluate_summary(machine_summary, gt_summary, video_name, score, eval_data=self.config.dataset)


        return kTau, sRho
    
    def test_epoch_end(self, outs):
        outs = torch.tensor(outs)

        kTau = outs[:,0].mean()
        sRho = outs[:,1].mean()

        self.log('val_kTau', kTau, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho', sRho, on_step=False, on_epoch=True, prog_bar=True)

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