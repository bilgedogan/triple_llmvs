import os
import torch
import argparse

from utils.configs import Config, str2bool
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
seed_everything(1112)
from pytorch_lightning.loggers import TensorBoardLogger
from networks.model import LLMVS

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment YAML (e.g. configs/experiments/summe_v1.yaml)')
    # CLI overrides — any value provided here wins over the YAML
    parser.add_argument('--split_idx',              type=int,   default=None)
    parser.add_argument('--epochs',                 type=int,   default=None)
    parser.add_argument('--lr',                     type=float, default=None)
    parser.add_argument('--reduced_dim',            type=int,   default=None)
    parser.add_argument('--input_dim',              type=int,   default=None)
    parser.add_argument('--hidden_dim',             type=int,   default=None)
    parser.add_argument('--num_model_layers',       type=int,   default=None)
    parser.add_argument('--num_mst_layers',         type=int,   default=None)
    parser.add_argument('--num_cmf_layers',         type=int,   default=None)
    parser.add_argument('--num_heads',              type=int,   default=None)
    parser.add_argument('--dropout',                type=float, default=None)
    parser.add_argument('--experiment_name',        type=str,   default=None)

    opt = parser.parse_args()
    overrides = {k: v for k, v in vars(opt).items() if k != 'config'}
    config = Config(config_path=opt.config, overrides=overrides)

    if config.dataset == 'summe':
        from utils.summe_dataset import SumMeLLaMADataset, TrainBatchCollator, ValBatchCollator
        train_dataset = SumMeLLaMADataset(mode='train', split_idx=config.split_idx, llama_embedding=config.pt_path)
        val_dataset   = SumMeLLaMADataset(mode='test',  split_idx=config.split_idx, llama_embedding=config.pt_path)
    elif config.dataset == 'tvsum':
        from utils.tvsum_dataset import TVSumLLaMADataset, TrainBatchCollator, ValBatchCollator
        train_dataset = TVSumLLaMADataset(mode='train', split_idx=config.split_idx, llama_embedding=config.pt_path)
        val_dataset   = TVSumLLaMADataset(mode='test',  split_idx=config.split_idx, llama_embedding=config.pt_path)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=TrainBatchCollator(), pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=ValBatchCollator(),   pin_memory=True, persistent_workers=True)

    model = LLMVS(config=config)
    model.cuda()

    best_rho_dir = f'{config.save_dir_root}/best_rho_model'
    best_tau_dir = f'{config.save_dir_root}/best_tau_model'

    checkpoint_rho = ModelCheckpoint(
        monitor='val_sRho',
        dirpath=best_rho_dir,
        filename='{epoch:02d}-{val_sRho:.3f}',
        save_top_k=1,
        save_last=True,
        mode='max',
    )
    checkpoint_tau = ModelCheckpoint(
        monitor='val_kTau',
        dirpath=best_tau_dir,
        filename='{epoch:02d}-{val_kTau:.3f}',
        save_top_k=1,
        save_last=True,
        mode='max',
    )
    checkpoint_f1 = ModelCheckpoint(
        monitor='val_f1',
        dirpath=f'{config.save_dir_root}/best_f1_model',
        filename='{epoch:02d}-{val_f1:.3f}',
        save_top_k=1,
        save_last=False,
        mode='max',
    )

    logger = TensorBoardLogger(
        save_dir='logs',
        name=config.experiment_name,
        version=f'split_{config.split_idx}',
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_rho, checkpoint_tau, checkpoint_f1],
        logger=logger,
        benchmark=True,
        deterministic=False,
        val_check_interval=0.5,
        progress_bar_refresh_rate=100,
        profiler='simple',
        log_every_n_steps=4,
    )

    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)
