import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.configs import Config
from utils.yaml_config import apply_yaml
from utils.multimodal_dataset import (
    MultimodalSummDataset,
    MultimodalTrainCollator,
    MultimodalValCollator,
)
from networks.multimodal_aggregator import PretrainPLModule

seed_everything(1112)


def _default_paths(dataset):
    if dataset == 'summe':
        return dict(
            llama_root='llama_emb/summe_sum',
            clip_path='clip_features/summe_clip.h5',
            audio_path='audio_features/summe_whisper.h5',
        )
    return dict(
        llama_root='llama_emb/tvsum_sum',
        clip_path='clip_features/tvsum_clip.h5',
        audio_path='audio_features/tvsum_whisper.h5',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config file (overrides defaults; CLI flags override YAML)')
    parser.add_argument('--exp_name', type=str, default='mm_pretrain_head2_layer3',
                        help='Experiment name — results land under Summaries/<exp_name>/<dataset>/<tag>/')
    parser.add_argument('--model', type=str, default=None, help='Deprecated alias for --exp_name')
    parser.add_argument('--dataset', type=str, default='summe')
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reduced_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--tag', type=str, default=None,
                        help='Sub-directory tag. Defaults to {dataset}_split{split_idx} if unset')
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--llama_root', type=str, default=None)
    parser.add_argument('--clip_path', type=str, default=None)
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    opt = parser.parse_args()
    apply_yaml(parser, opt, opt.config)

    if opt.model is None:
        opt.model = opt.exp_name
    if opt.tag is None:
        opt.tag = f'{opt.dataset}_split{opt.split_idx}'

    defaults = _default_paths(opt.dataset)
    for k, v in defaults.items():
        if getattr(opt, k) is None:
            setattr(opt, k, v)

    # Reuse Config so save_dir_root/configuration.txt logging stays consistent.
    config = Config(**vars(opt))

    train_ds = MultimodalSummDataset(
        dataset=opt.dataset, mode='train', split_idx=opt.split_idx,
        llama_root=opt.llama_root, clip_path=opt.clip_path, audio_path=opt.audio_path,
    )
    val_ds = MultimodalSummDataset(
        dataset=opt.dataset, mode='test', split_idx=opt.split_idx,
        llama_root=opt.llama_root, clip_path=opt.clip_path, audio_path=opt.audio_path,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False,
                              num_workers=opt.num_workers, collate_fn=MultimodalTrainCollator(),
                              pin_memory=True, persistent_workers=opt.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=opt.num_workers, collate_fn=MultimodalValCollator(),
                            pin_memory=True, persistent_workers=opt.num_workers > 0)

    model = PretrainPLModule(config)

    best_rho = f'{config.save_dir_root}/best_rho_model'
    best_tau = f'{config.save_dir_root}/best_tau_model'

    cb_rho = ModelCheckpoint(monitor='val_sRho', dirpath=best_rho,
                             filename='{epoch:02d}-{val_sRho:.3f}', save_top_k=1,
                             save_last=True, mode='max')
    cb_tau = ModelCheckpoint(monitor='val_kTau', dirpath=best_tau,
                             filename='{epoch:02d}-{val_kTau:.3f}', save_top_k=1,
                             save_last=True, mode='max')

    trainer = Trainer(
        gpus=1,
        max_epochs=opt.epochs,
        accumulate_grad_batches=2,
        precision=16,
        gradient_clip_val=0.01,
        callbacks=[cb_rho, cb_tau],
        benchmark=True,
        deterministic=False,
        val_check_interval=0.5,
        progress_bar_refresh_rate=100,
        log_every_n_steps=4,
    )

    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)
