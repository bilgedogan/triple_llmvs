"""Plotting + metric helpers for the multimodal RL pipeline (Phase 3 train / Phase 5 test).

Kept out of train_rl.py / test_rl.py so those stay lean. All figures use the Agg
backend (no display) and are written under the run's save_dir.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def f1_per_video(machine_summary, gt_summary, dataset):
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


def plot_training_curves(history, save_dir, fname='training_curves.png'):
    """Train vs val metrics over iters. Gap = overfit; both low = underfit."""
    its = history.get('iter', [])
    if not its:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ('train_tau', 'tab:blue', '-'),
        ('train_rho', 'tab:cyan', '-'),
        ('val_tau', 'tab:orange', '-'),
        ('val_rho', 'tab:green', '-'),
        ('val_f1', 'tab:red', '-'),
    ]
    for key, color, ls in series:
        ys = history.get(key)
        if ys and any(v is not None for v in ys):
            ax.plot(its, ys, ls, color=color, label=key)
    for key, color in (('best_tau', 'tab:orange'), ('best_rho', 'tab:green')):
        b = history.get(key)
        if b is not None and b > -1e8:
            ax.axhline(b, ls='--', lw=0.8, color=color, alpha=0.5,
                       label=f'{key}={b:.4f}')
    ax.set_xlabel('iter'); ax.set_ylabel('metric')
    ax.set_title('RL training curves')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)


def plot_loss_curves(history, save_dir, fname='loss_curves.png'):
    """PPO loss components over iters (policy / value / entropy / aux MSE)."""
    its = history.get('loss_iter', [])
    if not its:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color in (('pol', 'tab:blue'), ('val', 'tab:orange'),
                       ('ent', 'tab:green'), ('aux', 'tab:red')):
        ys = history.get(key)
        if ys:
            ax.plot(its, ys, color=color, label=key)
    ax.set_xlabel('iter'); ax.set_ylabel('loss')
    ax.set_title('PPO loss components')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)


def plot_metric_box(per_video, save_dir, fname='metric_box.png', title='per-video metrics'):
    """Box plot of per-video metric spread. `per_video` = dict name->list[float]."""
    items = [(k, v) for k, v in per_video.items() if v]
    if not items:
        return
    labels = [k for k, _ in items]
    data = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    for i, vals in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(vals))
        ax.plot(x, vals, '.', color='tab:gray', alpha=0.5, markersize=4)
    ax.set_ylabel('value'); ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)


def plot_score_comparison(video_name, pred_scores, gt_scores, save_dir, tag=''):
    """Predicted vs GT frame importance for one video."""
    pred = np.asarray(pred_scores, dtype=float).ravel()
    gt = np.asarray(gt_scores, dtype=float).ravel()
    L = min(len(pred), len(gt))
    pred, gt = pred[:L], gt[:L]
    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(x, gt, color='tab:green', alpha=0.3, label='GT importance')
    ax.plot(x, pred, color='tab:red', lw=1.2, label='predicted')
    ax.set_xlabel('frame'); ax.set_ylabel('importance')
    ax.set_title(f'{video_name} — predicted vs GT')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    safe = str(video_name).replace('/', '_')
    pref = f'scores_{tag}_' if tag else 'scores_'
    fig.savefig(os.path.join(save_dir, f'{pref}{safe}.png'), dpi=120)
    plt.close(fig)
