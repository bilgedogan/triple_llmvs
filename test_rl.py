import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from utils.yaml_config import apply_yaml
from utils.multimodal_dataset import (
    MultimodalSummDataset,
    MultimodalValCollator,
)
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary
from projections import FusionProjections, TextEncoder, CompressionProjections
from networks.multimodal_aggregator import MultimodalAggregator
from networks.rl_agent import RLAgent
from networks.lora import apply_lora_to_aggregator

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


def _resolve_weights(template, split_idx):
    """Substitute {split} and resolve globs to a concrete checkpoint path."""
    if template is None:
        raise ValueError('Provide --weights or set weights in YAML config')
    path = template.replace('{split}', str(split_idx))
    if any(ch in path for ch in '*?['):
        hits = sorted(glob.glob(path))
        if not hits:
            raise FileNotFoundError(f'No checkpoint matches: {path}')
        return hits[-1]
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def deterministic_scores(agent, fusion, comp, text_encoder, aggregator, batch, device):
    """Forward pass at test time: deterministic Dirichlet mode → fused → scores."""
    lu = batch['llama_user'][0].to(device)
    lg = batch['llama_gen'][0].to(device)
    txt = text_encoder(lu, lg)
    v = batch['visual'][0].to(device)
    a = batch['audio'][0].to(device)
    T = v.shape[0]

    h, c = agent.init_state(device)
    fused = []
    for t in range(T):
        v_t = v[t:t + 1]; a_t = a[t:t + 1]; txt_t = txt[t:t + 1]
        v_s, txt_s, a_s = comp(v_t, txt_t, a_t)
        norms_t = torch.stack([v[t].norm(), txt[t].norm(), a[t].norm()])
        t_norm = torch.tensor(t / max(T - 1, 1), device=device, dtype=v.dtype)
        alpha, _v, h, c, _ = agent.step(v_s, txt_s, a_s, h, c, t_norm, norms_t)
        w = alpha / alpha.sum(dim=-1, keepdim=True)
        v_f = fusion.project_visual(v_t); a_f = fusion.project_audio(a_t); txt_f = fusion.project_text(txt_t)
        f_t = w[:, 0:1] * v_f + w[:, 1:2] * txt_f + w[:, 2:3] * a_f
        fused.append(f_t.squeeze(0))
    F_fused = torch.stack(fused, dim=0).unsqueeze(0)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    scores = aggregator(F_fused, mask=mask).squeeze(0).clamp(0.0, 1.0)
    return scores


def _f1_per_video(machine_summary, gt_summary, dataset):
    """SumMe: max over user summaries. TVSum: mean over user summaries."""
    pred = machine_summary.detach().cpu().float()
    gts = gt_summary.detach().cpu().float()
    L = min(pred.shape[0], gts.shape[1])
    pred = pred[:L]; gts = gts[:, :L]
    f1_per = []
    for u in range(gts.shape[0]):
        tp = (pred * gts[u]).sum().item()
        pp = pred.sum().item(); pg = gts[u].sum().item()
        if pp == 0 or pg == 0:
            f1_per.append(0.0); continue
        prec = tp / pp; rec = tp / pg
        f1_per.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    if not f1_per:
        return 0.0
    return float(np.mean(f1_per)) if dataset == 'tvsum' else float(np.max(f1_per))


def _eval_split(opt, split_idx, weights_path, device, text_encoder):
    test_ds = MultimodalSummDataset(
        dataset=opt.dataset, mode='test', split_idx=split_idx,
        llama_root=opt.llama_root, clip_path=opt.clip_path, audio_path=opt.audio_path,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=opt.num_workers,
        collate_fn=MultimodalValCollator(), pin_memory=True,
    )

    fusion = FusionProjections(out_dim=opt.reduced_dim).to(device)
    comp = CompressionProjections(text_dim=opt.reduced_dim).to(device)
    aggregator = MultimodalAggregator(
        reduced_dim=opt.reduced_dim, num_heads=opt.num_heads, num_layers=opt.num_layers,
    ).to(device)
    agent = RLAgent().to(device)

    ckpt = torch.load(weights_path, map_location='cpu')
    has_lora = any('parametrizations' in k for k in ckpt['aggregator'].keys())
    if has_lora:
        apply_lora_to_aggregator(aggregator, rank=opt.lora_rank, alpha=opt.lora_alpha)

    agent.load_state_dict(ckpt['agent'])
    fusion.load_state_dict(ckpt['fusion'])
    comp.load_state_dict(ckpt['comp'])
    aggregator.load_state_dict(ckpt['aggregator'])
    agent.eval(); fusion.eval(); comp.eval(); aggregator.eval()

    taus, rhos, f1s = [], [], []
    summary_size = 0.15
    with torch.no_grad():
        for batch in test_loader:
            scores = deterministic_scores(agent, fusion, comp, text_encoder, aggregator, batch, device)
            cps = batch['change_points'][0]
            n_frames = batch['n_frames'][0]
            nfps = batch['n_frame_per_seg'][0].tolist()
            picks = batch['picks'][0]
            gt_summary = batch['gt_summary'][0]
            video_name = batch['video_name'][0]
            machine_summary = generate_summary(
                scores, cps, n_frames.unsqueeze(0), nfps, picks, proportion=summary_size,
            )
            kTau, sRho = evaluate_summary(
                machine_summary, gt_summary, video_name, scores, eval_data=opt.dataset,
            )
            taus.append(float(kTau)); rhos.append(float(sRho))
            f1s.append(_f1_per_video(machine_summary, gt_summary, opt.dataset))

    return float(np.mean(taus)), float(np.mean(rhos)), float(np.mean(f1s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    parser.add_argument('--exp_name', type=str, default='mm_rl_head2_layer3',
                        help='Experiment name — Summaries/<exp_name>/<dataset>/')
    parser.add_argument('--model', type=str, default=None, help='Deprecated alias for --exp_name')
    parser.add_argument('--dataset', type=str, default='summe', choices=['summe', 'tvsum'])
    parser.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--split_idx', type=int, default=None,
                        help='Run a single split (overrides --splits when set).')
    parser.add_argument('--reduced_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--weights', type=str, default=None,
                        help='Checkpoint template; {split} substituted, globs allowed.')
    parser.add_argument('--llama_root', type=str, default=None)
    parser.add_argument('--clip_path', type=str, default=None)
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--joint_finetune', action='store_true',
                        help='Set when the checkpoint was produced with --joint_finetune')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Where results.txt is written. '
                             'Defaults to Summaries/<exp_name>/<dataset>.')
    parser.add_argument('--result_file', type=str, default='results.txt')
    parser.add_argument('--num_workers', type=int, default=2)
    opt = parser.parse_args()
    apply_yaml(parser, opt, opt.config)

    if opt.model is None:
        opt.model = opt.exp_name
    else:
        opt.exp_name = opt.model

    defaults = _default_paths(opt.dataset)
    for k, v in defaults.items():
        if getattr(opt, k) is None:
            setattr(opt, k, v)

    if opt.split_idx is not None:
        opt.splits = [opt.split_idx]

    if opt.result_dir is None:
        opt.result_dir = f'Summaries/{opt.exp_name}/{opt.dataset}'
    os.makedirs(opt.result_dir, exist_ok=True)
    out_path = os.path.join(opt.result_dir, opt.result_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    text_encoder = TextEncoder(out_dim=opt.reduced_dim).to(device).eval()

    rows, taus, rhos, f1s = [], [], [], []
    for s in opt.splits:
        weights_path = _resolve_weights(opt.weights, s)
        kTau, sRho, F1 = _eval_split(opt, s, weights_path, device, text_encoder)
        taus.append(kTau); rhos.append(sRho); f1s.append(F1)
        line = (f'split {s}: kTau={kTau:.4f} sRho={sRho:.4f} F1={F1:.4f} '
                f'weights={weights_path}')
        print(line)
        rows.append(line)

    mean_line = (f'mean over {len(opt.splits)} splits: '
                 f'kTau={np.mean(taus):.4f} sRho={np.mean(rhos):.4f} F1={np.mean(f1s):.4f}')
    print(mean_line)

    with open(out_path, 'w') as f:
        f.write(f'exp_name={opt.exp_name} dataset={opt.dataset} '
                f'reduced_dim={opt.reduced_dim} num_heads={opt.num_heads} '
                f'num_layers={opt.num_layers} joint_finetune={opt.joint_finetune}\n')
        for r in rows:
            f.write(r + '\n')
        f.write(mean_line + '\n')
    print(f'[test] wrote {out_path}')


if __name__ == '__main__':
    main()
