import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from utils.configs import Config
from utils.yaml_config import apply_yaml
from utils.multimodal_dataset import (
    MultimodalSummDataset,
    MultimodalTrainCollator,
    MultimodalValCollator,
)
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary
from projections import (
    FusionProjections, TextEncoder, CompressionProjections,
)
from networks.multimodal_aggregator import MultimodalAggregator
from networks.rl_agent import RLAgent
from networks.lora import apply_lora_to_aggregator, lora_parameters


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


def _kendall_tau(pred, gt):
    p = pred.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    if np.std(p) < 1e-8 or np.std(g) < 1e-8:
        return 0.0
    tau, _ = stats.kendalltau(stats.rankdata(-p), stats.rankdata(-g))
    return float(tau) if not np.isnan(tau) else 0.0


def _diversity_reward(visual, scores, top_frac=0.15):
    T = visual.shape[0]
    k = max(2, int(round(top_frac * T)))
    top = torch.topk(scores, k=min(k, T)).indices
    v = F.normalize(visual[top], dim=-1)
    sim = v @ v.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return -sim[mask].mean()  # higher = more diverse


def _gae(rewards, values, gamma=1.0, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = 0.0 if t == T - 1 else values[t + 1].item()
        # temporal diff error: how much better is the reward at this step than expected
        # values is the critic's prediction, rewards is the actual current reward, next rewrd ecpected by critic, gamma discount factor
        delta = rewards[t].item() + gamma * next_v - values[t].item()
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values.detach()
    return advantages, returns


class FeatureCache:
    """Per-video cache of raw modality features (CPU tensors)."""

    def __init__(self):
        self.cache = {}

    def get(self, batch, text_encoder, device):
        name = batch['video_name'][0]
        if name not in self.cache:
            with torch.no_grad():
                lu = batch['llama_user'][0].to(device)
                lg = batch['llama_gen'][0].to(device)
                txt = text_encoder(lu, lg).cpu()
            self.cache[name] = dict(
                visual=batch['visual'][0],
                audio=batch['audio'][0],
                text=txt,
                pool5=batch['pool5'][0],
                gtscore=batch['gtscore'][0],
            )
        return self.cache[name]


def run_rollout(agent, fusion, comp, entry, device, fixed_weights=None):
    """Single-episode forward through compression → LSTM → actor → critic and
    optionally weighted fusion → fused sequence.

    When ``fixed_weights`` is provided (T, 3), use those weights instead of
    sampling — used during PPO re-rolls to keep actions consistent while
    refreshing gradients.

    Returns dict with tensors (gradient-enabled where appropriate):
        alphas (T,3), values (T,), log_probs (T,), weights (T,3),
        F_fused (1,T,2048), smooth_rewards (T,)
    """
    v = entry['visual'].to(device)
    a = entry['audio'].to(device)
    txt = entry['text'].to(device)
    T = v.shape[0]

    v_norms = v.norm(dim=-1)
    txt_norms = txt.norm(dim=-1)
    a_norms = a.norm(dim=-1)

    h, c = agent.init_state(device)
    alphas, values, log_probs, weights_out, fused = [], [], [], [], []
    smooth = []
    prev_w = None

    for t in range(T):
        v_t = v[t:t + 1]
        a_t = a[t:t + 1]
        txt_t = txt[t:t + 1]
        v_small, txt_small, a_small = comp(v_t, txt_t, a_t)
        norms_t = torch.stack([v_norms[t], txt_norms[t], a_norms[t]])
        t_norm = torch.tensor(t / max(T - 1, 1), device=device, dtype=v.dtype)
        alpha, value, h, c, _state = agent.step(
            v_small, txt_small, a_small, h, c, t_norm, norms_t,
        )
        dist = Dirichlet(alpha)
        if fixed_weights is None:
            w = dist.rsample()
        else:
            w = fixed_weights[t].unsqueeze(0)
        log_prob = dist.log_prob(w)

        v_fused = fusion.project_visual(v_t)
        a_fused = fusion.project_audio(a_t)
        txt_fused = fusion.project_text(txt_t)
        f_t = w[:, 0:1] * v_fused + w[:, 1:2] * txt_fused + w[:, 2:3] * a_fused

        alphas.append(alpha.squeeze(0))
        values.append(value.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        weights_out.append(w.squeeze(0))
        fused.append(f_t.squeeze(0))
        if prev_w is None:
            smooth.append(torch.zeros((), device=device))
        else:
            smooth.append(-(w.squeeze(0) - prev_w.detach()).pow(2).sum())
        prev_w = w.squeeze(0)

    return dict(
        alphas=torch.stack(alphas, dim=0),
        values=torch.stack(values, dim=0),
        log_probs=torch.stack(log_probs, dim=0),
        weights=torch.stack(weights_out, dim=0),
        F_fused=torch.stack(fused, dim=0).unsqueeze(0),
        smooth_rewards=torch.stack(smooth, dim=0),
        T=T,
    )


def compute_rewards(rollout, aggregator, entry, device, smoothness_coef, diversity_coef, correlation_coef=0.7):
    F_fused = rollout['F_fused']
    T = rollout['T']
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        scores = aggregator(F_fused, mask=mask).squeeze(0).clamp(0.0, 1.0)
    gt = entry['gtscore'].to(device)
    r1 = _kendall_tau(scores, gt)
    r2 = _diversity_reward(entry['pool5'].to(device), scores).item()

    # Map raw smoothness [-2.0, 0.0] -> [-1.0, 0.0]
    smooth_raw = rollout['smooth_rewards'].detach().clone()
    smooth_mapped = smooth_raw / 2.0

    # Map raw diversity [-1.0, 0.0] -> [-1.0, 1.0]
    r2_mapped = 2.0 * r2 + 1.0

    # Correlation is already in [-1.0, 1.0]
    r1_mapped = r1

    # Dense step-wise reward (normalized by sequence length T)
    rewards = (smoothness_coef * smooth_mapped) / T

    # Add sparse terminal rewards at the final step
    rewards[-1] = rewards[-1] + correlation_coef * r1_mapped + diversity_coef * r2_mapped
    return rewards, dict(r1=r1, r2=r2, ktau=r1, scores=scores)


def evaluate(agent, fusion, comp, text_encoder, aggregator, val_loader, device, dataset_name):
    agent.eval(); fusion.eval(); comp.eval(); aggregator.eval()
    cache = FeatureCache()
    taus, rhos = [], []
    with torch.no_grad():
        for batch in val_loader:
            entry = cache.get(batch, text_encoder, device)
            # deterministic: use alpha-mode as weights
            # quick path: do a rollout with sampling, then redo with mode for final scoring.
            v = entry['visual'].to(device); a = entry['audio'].to(device); txt = entry['text'].to(device)
            T = v.shape[0]
            h, h_c = agent.init_state(device)
            fused = []
            for t in range(T):
                v_t = v[t:t+1]; a_t = a[t:t+1]; txt_t = txt[t:t+1]
                v_s, txt_s, a_s = comp(v_t, txt_t, a_t)
                norms_t = torch.stack([v[t].norm(), txt[t].norm(), a[t].norm()])
                t_norm = torch.tensor(t / max(T-1, 1), device=device, dtype=v.dtype)
                alpha, _val, h, h_c, _state = agent.step(v_s, txt_s, a_s, h, h_c, t_norm, norms_t)
                w = alpha / alpha.sum(dim=-1, keepdim=True)
                v_f = fusion.project_visual(v_t); a_f = fusion.project_audio(a_t); txt_f = fusion.project_text(txt_t)
                fused.append((w[:,0:1]*v_f + w[:,1:2]*txt_f + w[:,2:3]*a_f).squeeze(0))
            F_fused = torch.stack(fused, dim=0).unsqueeze(0)
            mask = torch.ones(1, T, dtype=torch.bool, device=device)
            scores = aggregator(F_fused, mask=mask).squeeze(0).clamp(0.0, 1.0)
            cps = batch['change_points'][0]; n_frames = batch['n_frames'][0]
            nfps = batch['n_frame_per_seg'][0].tolist(); picks = batch['picks'][0]
            gt_summary = batch['gt_summary'][0]; video_name = batch['video_name'][0]
            machine_summary = generate_summary(scores, cps, n_frames.unsqueeze(0), nfps, picks)
            kTau, sRho = evaluate_summary(machine_summary, gt_summary, video_name, scores, eval_data=dataset_name)
            taus.append(float(kTau)); rhos.append(float(sRho))
    agent.train(); fusion.train(); comp.train()
    return float(np.mean(taus)), float(np.mean(rhos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    parser.add_argument('--exp_name', type=str, default='mm_rl_head2_layer3',
                        help='Experiment name — Summaries/<exp_name>/<dataset>/<tag>/')
    parser.add_argument('--model', type=str, default=None, help='Deprecated alias for --exp_name')
    parser.add_argument('--dataset', type=str, default='summe')
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200, help='PPO outer iterations')
    parser.add_argument('--episodes_per_update', type=int, default=8)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lora_lr', type=float, default=3e-5)
    parser.add_argument('--rl_lr_joint', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--reduced_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--tag', type=str, default=None,
                        help='Sub-directory tag. Defaults to {dataset}_split{split_idx}.')
    parser.add_argument('--llama_root', type=str, default=None)
    parser.add_argument('--clip_path', type=str, default=None)
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--pretrained_aggregator', type=str, default=None,
                        help='Phase 2 PretrainPLModule checkpoint (required, may be set via YAML)')
    parser.add_argument('--joint_finetune', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--smoothness_coef', type=float, default=0.2)
    parser.add_argument('--diversity_coef', type=float, default=0.1)
    parser.add_argument('--correlation_coef', type=float, default=0.7)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--aux_mse_coef', type=float, default=0.1,
                        help='Auxiliary MSE loss on aggregator scores — provides gradient '
                             'to fusion projections and LoRA params during PPO updates')
    parser.add_argument('--eval_every', type=int, default=1)
    opt = parser.parse_args()
    apply_yaml(parser, opt, opt.config)

    if opt.model is None:
        opt.model = opt.exp_name
    if opt.tag is None:
        opt.tag = f'{opt.dataset}_split{opt.split_idx}'
    if opt.pretrained_aggregator is None:
        raise ValueError('Provide --pretrained_aggregator or set pretrained_aggregator in YAML config')

    defaults = _default_paths(opt.dataset)
    for k, v in defaults.items():
        if getattr(opt, k) is None:
            setattr(opt, k, v)

    config = Config(**vars(opt))
    device = config.device

    train_ds = MultimodalSummDataset(
        dataset=opt.dataset, mode='train', split_idx=opt.split_idx,
        llama_root=opt.llama_root, clip_path=opt.clip_path, audio_path=opt.audio_path,
    )
    val_ds = MultimodalSummDataset(
        dataset=opt.dataset, mode='test', split_idx=opt.split_idx,
        llama_root=opt.llama_root, clip_path=opt.clip_path, audio_path=opt.audio_path,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=opt.num_workers, collate_fn=MultimodalTrainCollator(),
                              pin_memory=True, persistent_workers=opt.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=opt.num_workers, collate_fn=MultimodalValCollator(),
                            pin_memory=True, persistent_workers=opt.num_workers > 0)

    text_encoder = TextEncoder(out_dim=opt.reduced_dim).to(device)
    text_encoder.eval()
    fusion = FusionProjections(out_dim=opt.reduced_dim).to(device)
    comp = CompressionProjections(text_dim=opt.reduced_dim).to(device)
    aggregator = MultimodalAggregator(
        reduced_dim=opt.reduced_dim, num_heads=opt.num_heads, num_layers=opt.num_layers,
    ).to(device)
    agent = RLAgent().to(device)

    ckpt = torch.load(opt.pretrained_aggregator, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    agg_state = {k.replace('aggregator.', '', 1): v for k, v in state.items() if k.startswith('aggregator.')}
    fusion_state = {k.replace('fusion.', '', 1): v for k, v in state.items() if k.startswith('fusion.')}
    text_encoder_state = {k.replace('text_encoder.', '', 1): v for k, v in state.items() if k.startswith('text_encoder.')}
    miss, unex = aggregator.load_state_dict(agg_state, strict=False)
    if fusion_state:
        fusion.load_state_dict(fusion_state, strict=False)
    if text_encoder_state:
        text_encoder.load_state_dict(text_encoder_state, strict=False)
    print(f'[load] aggregator missing={len(miss)} unexpected={len(unex)}')

    for p in aggregator.parameters():
        p.requires_grad_(False)
    aggregator.eval()

    lora_modules = None
    if opt.joint_finetune:
        lora_modules = apply_lora_to_aggregator(aggregator, rank=opt.lora_rank, alpha=opt.lora_alpha)
        rl_lr = opt.rl_lr_joint
    else:
        rl_lr = opt.lr

    rl_params = list(agent.parameters()) + list(fusion.parameters()) + list(comp.parameters())
    if lora_modules is not None:
        optimizer = torch.optim.Adam([
            {'params': rl_params, 'lr': rl_lr},
            {'params': list(lora_parameters(lora_modules)), 'lr': opt.lora_lr},
        ])
    else:
        optimizer = torch.optim.Adam(rl_params, lr=rl_lr)

    cache = FeatureCache()

    best_tau = -1e9
    best_rho = -1e9
    save_dir = config.save_dir_root
    os.makedirs(save_dir, exist_ok=True)

    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    mse = torch.nn.MSELoss()

    for it in range(opt.epochs):
        t0 = time.time()
        # Phase A: collect rollouts (no grad on critic/actor — store data only).
        episodes = []
        train_taus = []
        for _ in range(opt.episodes_per_update):
            batch = next_batch()
            entry = cache.get(batch, text_encoder, device)
            with torch.no_grad():
                roll = run_rollout(agent, fusion, comp, entry, device)
                rewards, metrics = compute_rewards(
                    roll, aggregator, entry, device,
                    smoothness_coef=opt.smoothness_coef,
                    diversity_coef=opt.diversity_coef,
                    correlation_coef=opt.correlation_coef,
                )
                adv, ret = _gae(rewards, roll['values'])
            # Normalise advantage per-episode (small T means batch-wide normalisation is noisy).
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            episodes.append(dict(
                entry=entry,
                weights=roll['weights'].detach(),
                old_log_probs=roll['log_probs'].detach(),
                advantages=adv.detach(),
                returns=ret.detach(),
            ))
            train_taus.append(metrics['ktau'])

        # Phase B: PPO update — re-roll each epoch so encoder/LSTM/projections receive gradient.
        total_pol, total_val, total_ent, total_aux = 0.0, 0.0, 0.0, 0.0
        for _ in range(opt.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            n_frames = 0
            for ep in episodes:
                entry = ep['entry']
                roll = run_rollout(agent, fusion, comp, entry, device, fixed_weights=ep['weights'])
                new_log_probs = roll['log_probs']
                values = roll['values']
                alphas = roll['alphas']
                ratio = torch.exp(new_log_probs - ep['old_log_probs'])
                clipped = torch.clamp(ratio, 1 - opt.clip, 1 + opt.clip)
                pol = -torch.min(ratio * ep['advantages'], clipped * ep['advantages']).mean()
                val = (values - ep['returns']).pow(2).mean()
                ent = -Dirichlet(alphas).entropy().mean()
                # Auxiliary differentiable supervision so fusion projections + LoRA learn.
                T = roll['T']
                mask = torch.ones(1, T, dtype=torch.bool, device=device)
                scores = aggregator(roll['F_fused'], mask=mask).squeeze(0).clamp(0.0, 1.0)
                aux = mse(scores, entry['gtscore'].to(device))
                ep_loss = pol + opt.value_coef * val + opt.entropy_coef * ent + opt.aux_mse_coef * aux
                ep_loss = ep_loss * T  # weight by length so longer videos count proportionally
                loss_total = loss_total + ep_loss
                n_frames += T
                total_pol += pol.item(); total_val += val.item(); total_ent += ent.item(); total_aux += aux.item()
            (loss_total / max(n_frames, 1)).backward()
            params = [p for grp in optimizer.param_groups for p in grp['params']]
            torch.nn.utils.clip_grad_norm_(params, opt.grad_clip)
            optimizer.step()

        denom = max(opt.ppo_epochs * len(episodes), 1)
        dt = time.time() - t0
        print(f'[iter {it:04d}] train_kTau={np.mean(train_taus):.4f} '
              f'pol={total_pol/denom:.4f} val={total_val/denom:.4f} '
              f'ent={total_ent/denom:.4f} aux={total_aux/denom:.4f} ({dt:.1f}s)')

        if (it + 1) % opt.eval_every == 0:
            val_tau, val_rho = evaluate(agent, fusion, comp, text_encoder, aggregator,
                                        val_loader, device, opt.dataset)
            print(f'[iter {it:04d}] VAL kTau={val_tau:.4f} sRho={val_rho:.4f}')

            def _save(path, tau, rho):
                torch.save({
                    'agent': agent.state_dict(),
                    'fusion': fusion.state_dict(),
                    'comp': comp.state_dict(),
                    'aggregator': aggregator.state_dict(),
                    'text_encoder': text_encoder.state_dict(),
                    'iter': it, 'val_tau': tau, 'val_rho': rho,
                    'opt': vars(opt),
                }, path)

            if val_tau > best_tau:
                best_tau = val_tau
                _save(os.path.join(save_dir, 'best_tau.pt'), val_tau, val_rho)
            if val_rho > best_rho:
                best_rho = val_rho
                _save(os.path.join(save_dir, 'best_rho.pt'), val_tau, val_rho)


if __name__ == '__main__':
    main()
