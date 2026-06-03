# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n llmvs python=3.8
conda activate llmvs
pip install pip==23.3.2
conda install hdf5=1.10.6 h5py=2.10.0
pip install -r requirements.txt
```

Key pinned versions: `torch==1.13.1+cu117`, `pytorch-lightning==1.5.10`. PL version matters — API differences vs. newer versions (`validation_epoch_end`, `DDPPlugin`, `gpus=-1`).

## Two Pipelines

This repo contains two training pipelines that share datasets but are otherwise separate. Branch `main` (and `train.py` / `test.py`) is the original LLMVS paper code. Branch `rl_modality` adds a multimodal RL pipeline driven by YAML configs under `configs/` and shell launchers under `scripts/`.

### Pipeline A — legacy LLMVS (single-modality, Phase 1)

- Entry points: `train.py`, `test.py`, `test_splits.py`, `train.sh`, `test.sh`, `train_summe.sh`, `train_tvsum.sh`, `test_summe.sh`, `test_tvsum.sh`.
- Model: `networks/model.py::LLMVS` (PyTorch Lightning). Inputs are two LLaMA embeddings per frame (`user_prompt`, `gen`, both 5120-dim), concatenated → channel max-pool → linear → Transformer encoder (`num_layers` × `num_heads`) → 5-layer MLP → sigmoid frame scores.
- Hyperparameters per dataset: SumMe `lr=0.000119`, TVSum `lr=0.00007`. Default `--reduced_dim 2048 --num_heads 2 --num_layers 3 --epochs 200`. 5-fold CV (`--split_idx 0..4`).
- Loss MSE vs `gtscore`. Metrics Kendall τ (`val_kTau`), Spearman ρ (`val_sRho`).

Single split:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
  --tag summe_split0 --model summe_head2_layer3 \
  --lr 0.000119 --epochs 200 --dataset summe \
  --reduced_dim 2048 --num_heads 2 --num_layers 3 \
  --split_idx 0 --pt_path 'llama_emb/summe_sum/'

CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset summe --split_idx 0 --tag summe_split0 \
  --weights 'Summaries/summe_head2_layer3/summe/summe_split0/best_rho_model/<ckpt>.ckpt' \
  --pt_path llama_emb/summe_sum/ \
  --result_dir 'Summaries/summe_head2_layer3/summe/' \
  --num_heads 2 --num_layers 3 --reduced_dim 2048
```

### Pipeline B — multimodal RL (Phases 2–5)

Driven by YAML configs (`configs/*.yaml`) executed via `scripts/run_*.sh`. Each launcher reads the `splits` and `cuda_devices` keys from YAML, then loops over splits substituting `{split}` into checkpoint paths.

```bash
# Phase 2 — pretrain multimodal aggregator (5 folds)
bash scripts/run_pretrain.sh configs/pretrain_summe.yaml
# Phase 3 — PPO RL on top of pretrain ckpt (5 folds; YAML `pretrained_aggregator` uses {split})
bash scripts/run_rl.sh       configs/rl_summe_finetune.yaml
# Phase 5 — test (test_rl.py loops over splits internally)
bash scripts/run_test.sh     configs/test_summe.yaml
```

Phase 4 (joint LoRA fine-tuning of the frozen aggregator) is **toggled inside the Phase 3 config** via `joint_finetune: true` + `lora_rank` / `lora_alpha` / `lora_lr` / `rl_lr_joint`. There is no separate Phase 4 script.

CLI flags override YAML; both are merged in `utils/yaml_config.py::apply_yaml`. `scripts/_yaml_get.py` is a tiny YAML reader used only by the bash launchers.

## Architecture (Pipeline B)

Five-phase pipeline turning three modalities (visual, text, audio) into frame importance scores. Dimensions live in `projections.py`:

```
VISUAL_DIM=1024  AUDIO_DIM=512  FUSED_DIM=2048  COMP_DIM=256
```

1. **Feature inputs** (`utils/multimodal_dataset.py::MultimodalSummDataset`): per video, loads
   - Visual: CLIP features (`clip_features/{dataset}_clip.h5`, 1024-d/frame) + GoogleNet Pool5 (kept for legacy/diversity reward, in `SumMe/TVSum/*.h5`).
   - Audio: Whisper features (`audio_features/{dataset}_whisper.h5`, 512-d/frame).
   - Text: max-pooled LLaMA-3 embeddings (`llama_emb/{dataset}_sum/user_prompt/user_prompt_pool.h5` and `gen/gen_pool.h5`, 5120-d/frame each, raw).
   - `_align_lengths` enforces equal frame count across modalities (truncate to min).
2. **Projections** (`projections.py`):
   - `TextEncoder(llama_user, llama_gen) → FUSED_DIM=2048` (concat + MLP).
   - `FusionProjections.project_{visual,audio,text}(.) → FUSED_DIM` (per-modality 2048-d for fusion).
   - `CompressionProjections(v, txt, a) → 3× COMP_DIM=256` (low-dim features fed to RL agent).
3. **Phase 2 — `MultimodalAggregator`** (`networks/multimodal_aggregator.py`): Transformer encoder over fused features (`FUSED_DIM`, `num_heads`, `num_layers`) + MLP head → per-frame score in [0,1]. Pretrained with MSE vs `gtscore` under `PretrainPLModule` (Lightning). Saved under `Summaries/<exp_name>/<dataset>/<dataset>_split<idx>/best_{tau,rho}_model/`.
4. **Phase 3 — PPO** (`train_rl.py`): an `RLAgent` (`networks/rl_agent.py`, LSTM + Actor/Critic) emits per-frame Dirichlet weights over the 3 modalities. Weighted sum of `FUSED_DIM` projections becomes the fused stream fed to the (frozen) `MultimodalAggregator`. Rewards (`compute_rewards`):
   - Terminal: `r1 = Kendall τ(scores, gt)` + `diversity_coef · r2` (visual diversity of top-15% frames).
   - Per-frame: `smoothness_coef · -(w_t - w_{t-1})²` to dampen modality-weight oscillation.
   - Loss: PPO clipped policy + `value_coef · MSE(V, return)` + `entropy_coef · -H(Dirichlet)` + `aux_mse_coef · MSE(scores, gt)`.
   - GAE with `gamma=1.0, lam=0.95`. Checkpoints `best_tau.pt` / `best_rho.pt` saved under `Summaries/<exp_name>/<dataset>/<tag>/`.
5. **Phase 4 — LoRA joint fine-tune** (`networks/lora.py::apply_lora_to_aggregator`): wraps every `nn.Linear` weight in the aggregator with a low-rank delta (`rank`, `alpha`). When `--joint_finetune` is set, train_rl.py unfreezes aggregator LoRA params and lowers the RL LR to `rl_lr_joint` while LoRA params get `lora_lr`.
6. **Phase 5 — Test** (`test_rl.py`): rebuilds agent + projections + aggregator, loads `weights` (YAML template with `{split}`), runs `deterministic_scores` (uses Dirichlet *mode* = `alpha / Σα`, no sampling), then `evaluate_summary` over user annotations. Writes `results.txt` with mean τ/ρ across splits.

## Data Flow

- Splits: `dataset/{summe,tvsum}_splits.json` (5-fold key lists).
- Backbone HDF5s: `SumMe/eccv16_dataset_summe_google_pool5.h5`, `TVSum/eccv16_dataset_tvsum_google_pool5.h5` (Pool5 + `gtscore` + `user_summary`).
- TVSum eval reads raw `TVSum/ydata-tvsum50-anno.tsv` (1-based `video_<i>` index parsed from key suffix) for per-annotator correlations. SumMe uses the mean user summary from the H5.
- Checkpoint layout:
  - Pipeline A: `Summaries/<model>/<dataset>/<tag>/best_{rho,tau}_model/*.ckpt` (PL).
  - Pipeline B Phase 2: same Lightning layout via `PretrainPLModule`.
  - Pipeline B Phase 3/4: plain `torch.save` under `Summaries/<exp_name>/<dataset>/<tag>/best_{tau,rho}.pt`.
- Score → summary: `utils/generate_summary.py` aggregates frame scores into segment scores then knapsack-selects at 15% budget; `utils/evaluation_metrics.py::evaluate_summary` returns F-score, τ, ρ.

## Working with Configs

- Phase 2 configs (`configs/pretrain_*.yaml`) set `exp_name`, `splits`, `lr`, modality paths.
- Phase 3 configs (`configs/rl_*.yaml`, `rl_*_finetune*.yaml`) add PPO knobs, reward coefs, LoRA params, and a `pretrained_aggregator` template with `{split}`.
- Phase 5 configs (`configs/test_*.yaml`) set a `weights` template with `{split}` pointing at `best_tau.pt` (or `best_rho.pt`).
- The `{split}` placeholder is substituted by the launcher (`run_rl.sh`) or by `test_rl.py` directly.
- `tag` defaults to `{dataset}_split{split_idx}` if unset — preserves the 5-fold directory layout.

## Tuning Notes (from prior runs)

- SumMe has ~20 train videos/split, so it is more sensitive than TVSum (~40) to high `rl_lr_joint` and low `lora_rank` — small LoRA + high joint LR causes regressions on SumMe even when TVSum improves with the same config.
- Reward `r1` (τ) only fires on the terminal step; smoothness/diversity coefs accumulate across all frames, so over-weighting them can drown the τ signal.
- `aux_mse_coef` anchors PPO output near the pretrain target — raising it stabilizes training on small splits.

## TVSum Evaluation Quirk

TVSum scoring uses raw TSV (`TVSum/ydata-tvsum50-anno.tsv`) at eval; `evaluate_summary` parses it directly to compute per-annotator correlations. Video index is 1-based, parsed from the `video_name` suffix (e.g. `video_1`). SumMe uses the mean user summary from the H5 file.
