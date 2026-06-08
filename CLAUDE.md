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

Key pinned versions: `torch==1.13.1+cu117`, `pytorch-lightning==1.5.10`. PL version matters — API differs vs. newer versions (`validation_epoch_end`, `gpus=`, `progress_bar_refresh_rate`).

## Two Pipelines

Repo holds two training pipelines sharing datasets but otherwise separate. Branch `main` (`train.py` / `test.py`) is original LLMVS paper code. Branch `rl_modality` adds multimodal RL pipeline driven by YAML configs (`configs/`) + shell launchers (`scripts/`).

### Pipeline A — legacy LLMVS (single-modality)

- Entry points: `train.py`, `test.py`, `test_splits.py`, `train.sh`, `test.sh`, `train_{summe,tvsum}.sh`, `test_{summe,tvsum}.sh`.
- Model: `networks/model.py::LLMVS` (Lightning). Two LLaMA embeddings/frame (`user_prompt`, `gen`, 5120-d) → channel max-pool → linear → Transformer encoder → 5-layer MLP → sigmoid frame scores.
- Per-dataset hyperparams: SumMe `lr=0.000119`, TVSum `lr=0.00007`. Defaults `--reduced_dim 2048 --num_heads 2 --num_layers 3 --epochs 200`. 5-fold CV (`--split_idx 0..4`).
- Loss MSE vs `gtscore`. Metrics Kendall τ (`val_kTau`), Spearman ρ (`val_sRho`).

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --tag summe_split0 --model summe_head2_layer3 \
  --lr 0.000119 --epochs 200 --dataset summe \
  --reduced_dim 2048 --num_heads 2 --num_layers 3 \
  --split_idx 0 --pt_path 'llama_emb/summe_sum/'
```

### Pipeline B — multimodal RL

Driven by YAML configs (`configs/*.yaml`) run via `scripts/run_*.sh`. Each launcher reads `splits` + `cuda_devices` from YAML, loops over splits substituting `{split}` into checkpoint paths.

```bash
# Phase 2 — pretrain multimodal aggregator (5 folds)
bash scripts/run_pretrain.sh configs/pretrain_summe.yaml
# Phase 3 — PPO RL on top of pretrain ckpt (YAML `pretrained_aggregator` uses {split})
bash scripts/run_rl.sh       configs/rl_summe_finetune_v4.yaml
# Phase 5 — test (test_rl.py loops over splits internally)
bash scripts/run_test.sh     configs/test_summe.yaml
```

Run a single phase/split directly (bypassing the launcher):
```bash
python pretrain_aggregator.py --config configs/pretrain_summe.yaml --split_idx 0
python train_rl.py --config configs/rl_summe_finetune_v4.yaml --split_idx 0 \
  --pretrained_aggregator Summaries/mm_pretrain_summe_v2/summe/summe_split0/best_tau_model/last.ckpt
python test_rl.py --config configs/test_summe.yaml --split_idx 0
```

Phase 4 (joint LoRA fine-tune of the frozen aggregator) is **toggled inside the Phase 3 config** via `joint_finetune: true` + `lora_rank`/`lora_alpha`/`lora_lr`/`rl_lr_joint`. No separate Phase 4 script.

Config precedence: CLI flags > YAML > argparse defaults, merged in `utils/yaml_config.py::apply_yaml`. `scripts/_yaml_get.py` is a tiny YAML reader used only by the bash launchers (`splits`, `cuda_devices`, `pretrained_aggregator` keys).

## Architecture (Pipeline B)

Five-phase pipeline turning three modalities (visual, text, audio) into frame importance scores. Dims in `projections.py`:

```
VISUAL_DIM=1024  AUDIO_DIM=512  TEXT_RAW_DIM=5120  FUSED_DIM=2048  COMP_DIM=256
```

1. **Feature inputs** (`utils/multimodal_dataset.py::MultimodalSummDataset`): per video loads
   - Visual: CLIP (`clip_features/{dataset}_clip.h5`, 1024-d/frame) + GoogLeNet Pool5 (`SumMe/TVSum/*.h5`, kept only for diversity reward).
   - Audio: Whisper (`audio_features/{dataset}_whisper.h5`, 512-d/frame).
   - Text: raw LLaMA-3 embeddings (`llama_emb/{dataset}_sum/{user_prompt,gen}/*_pool.h5`, 5120-d/frame, shape `(T, tokens, 5120)`).
   - `_align_lengths` truncates all modalities to common min frame count. Batch size is always 1 (variable-length CV).
2. **Projections** (`projections.py`):
   - `TextEncoder(llama_user, llama_gen) → 2048` (concat over tokens → max-pool → linear → LayerNorm).
   - `FusionProjections.project_{visual,audio,text}(.) → 2048`, L2-normalised (per-modality fusion stream).
   - `CompressionProjections(v, txt, a) → 3× 256` (low-dim features fed to RL agent state).
3. **Phase 2 — `MultimodalAggregator`** (`networks/multimodal_aggregator.py`): Transformer encoder over fused 2048-d features + MLP head → per-frame score in [0,1]. Pretrained with masked MSE vs `gtscore` under `PretrainPLModule` (Lightning, equal-weight fusion, `precision=16`, cosine LR). Saved `Summaries/<exp_name>/<dataset>/<dataset>_split<idx>/best_{tau,rho}_model/`.
4. **Phase 3 — PPO** (`train_rl.py`): `RLAgent` (`networks/rl_agent.py`: `LSTMCell` + `StateEncoder` + Dirichlet `Actor` + `Critic`) emits per-frame Dirichlet concentrations over the 3 modalities. Sampled weights `w` form fused stream `w·[v,txt,a]` fed to the **frozen** aggregator. Aggregator stays frozen + `eval()` in Phase 3; only agent + fusion + comp (+ LoRA in Phase 4) train. Reward shaping in `compute_rewards`:
   - Terminal step: `correlation_coef·τ(scores,gt) + diversity_coef·(2·r2+1)`, where `r2` = visual diversity of top-15% frames (mapped to [-1,1]).
   - Dense per-frame: `smoothness_coef·(smooth/2)/T`, `smooth = -(w_t - w_{t-1})²` (dampens modality-weight oscillation).
   - Loss: PPO clipped policy + `value_coef·MSE(V,return)` + `entropy_coef·-H(Dirichlet)` + `aux_mse_coef·MSE(scores,gt)`. The aux MSE is the only differentiable path that trains fusion projections + LoRA, since aggregator scores are otherwise computed under `no_grad` for reward.
   - GAE `gamma=1.0, lam=0.95`. Advantages normalised per-episode. PPO re-rolls each epoch with `fixed_weights` (replays sampled actions) to refresh gradients. Checkpoints `best_{tau,rho}.pt` under `Summaries/<exp_name>/<dataset>/<tag>/`.
5. **Phase 4 — LoRA joint fine-tune** (`networks/lora.py`): `apply_lora_to_aggregator` wraps **only each self-attention `in_proj_weight` + `out_proj.weight`** (not the MLP head) with an additive low-rank delta via `torch.nn.utils.parametrize`. Base weights stay frozen; only LoRA A/B train. `--joint_finetune` lowers RL LR to `rl_lr_joint` and gives LoRA params `lora_lr`.
6. **Phase 5 — Test** (`test_rl.py`): rebuilds agent + projections + aggregator, loads `weights` (YAML template with `{split}`, globs allowed). `deterministic_scores` uses Dirichlet **mode** (`alpha/Σα`, no sampling). Detects LoRA from checkpoint keys (`parametrizations`) and re-applies before loading. Reports per-split + mean τ/ρ/F1 to `results.txt`. F1 = max over user summaries (SumMe) / mean (TVSum).

## Data Flow & Checkpoints

- Splits: `dataset/{summe,tvsum}_splits.json` (5-fold key lists, `train_keys`/`test_keys`).
- Backbone HDF5: `SumMe/eccv16_dataset_summe_google_pool5.h5`, `TVSum/eccv16_dataset_tvsum_google_pool5.h5` (Pool5 + `gtscore` + `user_summary` + change points/picks).
- SumMe LLaMA keys resolved via `video_name` field in the H5 (`_resolve_llama_key_summe`); TVSum keys are the raw `video_<i>` string.
- Score → summary: `utils/generate_summary.py` aggregates frame→segment scores then knapsack-selects at 15% budget; `utils/evaluation_metrics.py::evaluate_summary` returns F-score, τ, ρ.
- Checkpoint layouts:
  - Pipeline A + Phase 2: Lightning `.ckpt` under `.../best_{rho,tau}_model/`.
  - Phase 3/4: plain `torch.save` dict (`agent`/`fusion`/`comp`/`aggregator`/`text_encoder`/`opt`) → `best_{tau,rho}.pt`.

## Working with Configs

- `configs/pretrain_*.yaml`: `exp_name`, `splits`, `lr`, modality paths.
- `configs/rl_*.yaml`, `rl_*_finetune*.yaml` (versioned `_v2`/`_v3`/`_v4`): PPO knobs, reward coefs, LoRA params, `pretrained_aggregator` template with `{split}`.
- `configs/test_*.yaml`: `weights` template with `{split}` → `best_tau.pt` (or `best_rho.pt`).
- `tag` defaults to `{dataset}_split{split_idx}` — preserves 5-fold directory layout. `exp_name` (formerly `--model`, still a deprecated alias) is the top-level results dir.

## Tuning Notes (from prior runs)

- SumMe has ~20 train videos/split (TVSum ~40) — more sensitive to high `rl_lr_joint` + low `lora_rank`. Small LoRA + high joint LR causes SumMe regression even when TVSum improves with the same config. SumMe configs use higher `aux_mse_coef` (e.g. 2.0) to anchor PPO near the pretrain target.
- τ only fires on the terminal step; smoothness/diversity accumulate across all frames — over-weighting their coefs drowns the τ signal.

## TVSum Evaluation Quirk

TVSum scoring reads raw `TVSum/ydata-tvsum50-anno.tsv` at eval; `evaluate_summary` parses it for per-annotator correlations. Video index is 1-based, parsed from the `video_name` suffix (e.g. `video_1`). SumMe uses the mean user summary from the H5.
