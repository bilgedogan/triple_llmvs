import os
import torch
import argparse

from utils.configs import Config, str2bool
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
seed_everything(1112)
from networks.model import LLMVS

METRICS = [('val_f1','F1'),('val_kTau','kTau'),('val_sRho','sRho'),('val_mp15','mp15'),('val_map50','mAP50')]
KEYS    = [k for k,_ in METRICS]
LABELS  = [l for _,l in METRICS]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str, required=True)
    parser.add_argument('--weights',         type=str, required=True)
    parser.add_argument('--split_idx',       type=int, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--monitor',         type=str, default='rho')

    opt = parser.parse_args()
    overrides = {k: v for k, v in vars(opt).items() if k not in ('config', 'weights')}
    config = Config(config_path=opt.config, overrides=overrides)

    if config.dataset == 'summe':
        from utils.summe_dataset import SumMeLLaMADataset, ValBatchCollator
        test_dataset = SumMeLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding=config.pt_path)
    elif config.dataset == 'tvsum':
        from utils.tvsum_dataset import TVSumLLaMADataset, ValBatchCollator
        test_dataset = TVSumLLaMADataset(mode='test', split_idx=config.split_idx, llama_embedding=config.pt_path)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32,
                             collate_fn=ValBatchCollator(), pin_memory=True)

    model = LLMVS.load_from_checkpoint(opt.weights, config=config)
    model.cuda()
    model.eval()

    trainer = Trainer(
        gpus=1,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        benchmark=True,
        deterministic=False,
        progress_bar_refresh_rate=100,
        log_every_n_steps=1,
    )

    results = trainer.test(model, test_loader, ckpt_path=opt.weights)

    if results:
        m   = {k: float(v) for k, v in results[0].items()}
        C   = 9
        row = f"{config.split_idx:<6}" + ''.join(f"{m.get(k, float('nan')):>{C}.4f}" for k in KEYS)

        # Append split row to experiment-level best_{monitor}_results.txt
        exp_root = os.path.dirname(config.save_dir_root)
        out_path = os.path.join(exp_root, f'best_{opt.monitor}_results.txt')
        with open(out_path, 'a') as f:
            f.write(row + '\n')

        print(f"\n[split {config.split_idx}] " + "  ".join(f"{l}={m.get(k,float('nan')):.4f}" for k,l in METRICS))
