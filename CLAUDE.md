# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is an extension of **LLMVS** (Video Summarization with Large Language Models, CVPR25) into a **multimodal reinforcement-learning modality-fusion** framework (internally "triple_llmvs", active branch `rl_modality`). Two pipelines coexist:

1. **Original LLMVS (text-only)** — `train.py`, `test.py`, `test_splits.py`, `networks/model.py` (the `LLMVS` LightningModule), `utils/summe_dataset.py`, `utils/tvsum_dataset.py`. Operates only on LLaMA embeddings. Kept as the baseline; the README documents this pipeline.
2. **Multimodal RL pipeline (the current work)** — everything under `pretrain_aggregator.py`, `train_rl.py`, `test_rl.py`, `test_pretrain.py`, `projections.py`, `networks/{multimodal_aggregator,rl_agent,lora}.py`, `utils/multimodal_dataset.py`. Fuses three modalities: **visual** (CLIP, 1024d), **audio** (Whisper, 512d), and **text** (LLaMA user_prompt + gen, 5120d → 2048d). New code is what you'll usually be editing.

`version.py` is a hand-maintained changelog (newest block on top); commits are tagged `vX.Y.0` matching it. Bump it when making a release-worthy change.

## Environment

Python 3.8 / Torch 1.13.1+cu117 / PyTorch-Lightning **1.5.10** (the old Trainer API: `gpus=`, `progress_bar_refresh_rate`, `validation_epoch_end`, etc. — do not "modernize" to the new API). h5py/hdf5 must be conda-installed before pip:

```bash
conda create -n llmvs python=3.8 && conda activate llmvs
pip install pip==23.3.2
conda install hdf5=1.10.6 h5py=2.10.0
pip install -r requirements.txt
```

## Data layout (all gitignored; must exist locally)

| Dir | Contents |
|-----|----------|
| `SumMe/`, `TVSum/` | `eccv16_dataset_*_google_pool5.h5` (GoogLeNet pool5, gtscore, change_points, picks…) |
| `llama_emb/{summe,tvsum}_sum/` | `gen/gen_pool.h5`, `user_prompt/user_prompt_pool.h5` (raw 5120d LLaMA, max-pooled) |
| `clip_features/` | `{summe,tvsum}_clip.h5` (CLIP visual) |
| `audio_features/` | `{summe,tvsum}_whisper.h5` (Whisper audio) |
| `dataset/` | `{summe,tvsum}_splits.json` — 5-fold cross-validation key lists |

All training/eval uses **batch_size=1** (variable-length videos, 5-fold CV via `--split_idx 0..4`). `seed_everything(1112)` is set in every entrypoint.

## Multimodal pipeline: the phases

The pipeline is staged. Each stage has YAML configs in `configs/` and a launcher in `scripts/`.

- **Phase 2 — pretrain aggregator** (`pretrain_aggregator.py` → `PretrainPLModule`): trains `TextEncoder` + `FusionProjections` + `MultimodalAggregator` with **equal-weight** fusion (`fusion_mode: equal`) or `text_only`. Produces the Lightning checkpoints later consumed by RL.
- **Phase 3 — RL modality fusion** (`train_rl.py`): loads the Phase 2 aggregator + fusion **frozen**; trains the `RLAgent` (LSTM → actor emitting a **Dirichlet over the 3 modality weights**) + `CompressionProjections` via **REINFORCE with a global baseline** (mean of all step-returns; γ=1.0). No critic/PPO is used despite `Critic` existing in `rl_agent.py`. Per-frame modality weights `w` produce a weighted fused sequence scored by the frozen aggregator; reward = terminal `correlation_coef·τ + diversity_coef·diversity` + dense `smoothness_coef·smooth/T`. An auxiliary MSE (`aux_mse_coef`) gives a differentiable gradient to comp/agent.
- **Phase 4 — joint fine-tune** (`--joint_finetune` / `joint_finetune: true` in YAML): attaches **LoRA** (`networks/lora.py`, via `torch.nn.utils.parametrize`) to the aggregator's attention `in_proj`/`out_proj`; only LoRA A/B + RL params train.
- **Phase 5 — test**: `test_rl.py` (RL checkpoints, deterministic Dirichlet **mode** as weights) and `test_pretrain.py` (Phase 2 aggregator, equal-weight). Both iterate splits internally and write `results.txt` / `results_pretrain.txt` under `Summaries/<exp_name>/<dataset>/`.

## RL formulation (Phase 3 detail)

The Phase 3 agent learns **per-frame modality mixing weights**: at each video frame it decides how much to trust visual vs. text vs. audio before the frozen aggregator scores the fused sequence. One **episode = one full video**; the LSTM state resets per video (`RLAgent.init_state`).

### Networks and dims

| Module | File | Role | dims |
|--------|------|------|------|
| `TextEncoder` | `projections.py` | LLaMA user+gen → text feature | cat → token max-pool → `Linear(5120→2048)` + LN |
| `FusionProjections` | `projections.py` | per-modality → fused space for the **aggregator input** | visual `1024→2048`, audio `512→2048`, text identity (already 2048); each + LayerNorm |
| `CompressionProjections` | `projections.py` | per-modality → small **RL state** features | visual/audio/text → `256` each |
| `RLAgent` | `networks/rl_agent.py` | LSTM + StateEncoder + Actor (+ unused Critic) | see below |
| `MultimodalAggregator` | `networks/multimodal_aggregator.py` | fused `(B,T,2048)` → per-frame importance score `[0,1]` | TransformerEncoder + MLP head + Sigmoid |

`RLAgent` internals (`COMP_DIM=256`): `LSTMCell(768→256)` (input = the 3 compressed features concatenated); `StateEncoder` `Linear(1028→512)+ReLU`; `Actor` `512→256→256→3` (softplus + clamp); `Critic` `512→…→1` **exists but is never used** (no value loss, no PPO).

### State → action → fusion (per frame `t`, in `run_rollout`)

1. **State.** Compress the three raw features: `v_s, txt_s, a_s = comp(v_t, txt_t, a_t)` (each 256d). Step the LSTM on `cat(v_s, txt_s, a_s)` (768d). Build `state_raw = cat(v_s, txt_s, a_s, h_prev, t/T, [‖v‖,‖txt‖,‖a‖])` = **1028d**, then `state = StateEncoder(state_raw)` (512d). (Raw L2 norms and normalized timestep are explicit state features.)
2. **Action.** `alpha = Actor(state)` → 3 Dirichlet concentrations (`softplus + 1e-4`, clamped `[0.1, 20]`). Sample weights `w ~ Dirichlet(alpha)` via **`rsample()`** (reparameterized — this matters, see losses) and record `log_prob(w)`. `w` is a point on the 3-simplex; index order is **`[0]=visual, [1]=text, [2]=audio`**.
3. **Fusion.** `f_t = w₀·project_visual(v_t) + w₁·project_text(txt_t) + w₂·project_audio(a_t)` (fusion projections are **frozen**, loaded from Phase 2). Stack over `t` → `F_fused (1,T,2048)`.
4. **Score.** `scores = aggregator(F_fused).clamp(0,1)` — aggregator **frozen** (Phase 3) or LoRA-adapted (Phase 4).

### Reward (`compute_rewards`, computed under `no_grad`)

Three terms, all mapped to comparable ranges:

- **Correlation (terminal)** `r1` = Kendall τ between `scores` and `gtscore`, already in `[-1,1]`. Primary signal.
- **Diversity (terminal)** `r2` = `-mean cosine similarity` among the **top-15%** scored frames' `pool5` features (`∈[-1,0]`), mapped to `2·r2+1 ∈ [-1,1]`. Higher = the selected frames are more visually varied.
- **Smoothness (dense, per step)** `-‖wₜ − wₜ₋₁‖²` (`∈[-2,0]`), mapped `/2 → [-1,0]`. Penalizes jerky frame-to-frame weight changes.

Assembled: every step gets the dense `smoothness_coef · smooth_mapped / T`; the **final step additionally** gets `correlation_coef · r1 + diversity_coef · r2_mapped`. Returns are Monte-Carlo reward-to-go `Gₜ` with **γ=1.0** (`_returns`).

### Algorithm — REINFORCE with a global baseline

Per outer iteration (`epochs` = number of REINFORCE updates, **not** dataset passes):

1. Roll out `episodes_per_update` videos (sampled from the train loader), each rollout run **with grad**.
2. **Baseline** = mean over *every step-return of every episode* (one global scalar). `advantage = returns − baseline`, **detached**.
3. Per-episode loss:
   - `pol = -(log_probs · advantage).mean()` — REINFORCE score-function term.
   - `ent = -Dirichlet(alphas).entropy().mean()` — added as `entropy_coef·ent`, i.e. an **entropy bonus** (minimizing `ent` maximizes Dirichlet entropy → keeps exploration / avoids collapsing to one modality).
   - `aux = MSE(scores, gtscore)` — added as `aux_mse_coef·aux`, a **differentiable supervised anchor**.
   - `ep_loss = (pol + entropy_coef·ent + aux_mse_coef·aux) · T` (×T so longer videos count proportionally).
4. Sum episode losses, divide by total frame count, `backward()`, clip grads to `grad_clip`, `optimizer.step()`.

There is **no critic, no PPO ratio/clip, no GAE, no re-rollout.**

### Which loss trains which network

| Loss term | Gradient path | Trains |
|-----------|---------------|--------|
| `pol` (policy) | `log_prob` → Dirichlet → `alpha` → Actor → StateEncoder → (`v_s,txt_s,a_s`, LSTM) | **Actor, StateEncoder, LSTM, `comp`** |
| `entropy_coef·ent` | `alpha` → Actor → … | same as above (regularizes the policy) |
| `aux_mse_coef·aux` | `scores` → aggregator → `F_fused` → **`w` (reparameterized `rsample`)** → `alpha` → Actor → `comp`/LSTM; and aggregator weights | **Actor/`comp`/LSTM** via the reparameterized weight, plus **LoRA params** (Phase 4 only) |

Because `w` comes from `rsample`, the aux MSE backprops a low-variance pathwise gradient into the policy in addition to the high-variance REINFORCE term. **Frozen throughout Phase 3:** `fusion`, `aggregator` (base weights), `text_encoder` (eval, never in the optimizer). In **Phase 4**, LoRA A/B deltas on the aggregator attention become trainable (separate `lora_lr` param group) and `aux` is what supplies their gradient. The `Critic` receives no loss and is effectively dead.

### Evaluation / test-time

Deterministic: instead of sampling, use the Dirichlet **mode/mean** `w = alpha / alpha.sum()` as the weights (`evaluate` in `train_rl.py`, `deterministic_scores` in `test_rl.py`), then the same fuse → aggregator → `generate_summary` → τ/ρ/F1.

## Running things

Always go through the `scripts/*.sh` launchers with a config; they read `splits` and `cuda_devices` from the YAML, set `CUDA_VISIBLE_DEVICES`, loop folds, and substitute the `{split}` placeholder in checkpoint path templates.

```bash
bash scripts/run_pretrain.sh       configs/pretrain_summe_equal.yaml   # Phase 2
bash scripts/run_rl.sh             configs/rl_summe_equal_v1.yaml      # Phase 3 (+Phase 4 if joint_finetune)
bash scripts/run_test_pretrain.sh  configs/test_pretrain_summe.yaml    # Phase 5 (pretrain)
bash scripts/run_test.sh           configs/test_summe.yaml             # Phase 5 (RL)
```

Single fold directly (bypassing the loop):

```bash
python train_rl.py --config configs/rl_summe_equal_v1.yaml --split_idx 0
```

Original baseline (text-only): `bash train_summe.sh` / `bash test_summe.sh`, or `python train.py …` as in the README.

### Config precedence

`utils/yaml_config.apply_yaml` merges in this order (highest wins): **explicit CLI flags > YAML (`--config`) > argparse defaults**. So a flag the user typed always overrides the YAML. `scripts/_yaml_get.py` is the tiny shell-side YAML reader used by launchers.

## Output / checkpoint conventions

Everything lands under `Summaries/<exp_name>/<dataset>/<tag>/`, where `tag` defaults to `{dataset}_split{split_idx}`. `Config.set_dataset_dir` also writes `configuration.txt` there. Note `--model` is a **deprecated alias for `--exp_name`** — both still flow into the `Summaries/<model>/…` path.

- **Phase 2** saves Lightning `best_tau_model/` and `best_rho_model/` dirs (monitored on `val_kTau` / `val_sRho`).
- **Phase 3/4** save plain `torch.save` dicts `best_{tau,rho}_iter{NNNN}_{metric}{score}.pt` bundling `agent`/`fusion`/`comp`/`aggregator`/`text_encoder` state, plus `train_log.csv` and training-curve / loss / per-video box plots (`utils/visualize.py`). Only the current best per metric is kept (old ones globbed and deleted), so the test-side `{split}`-templated glob stays unambiguous.

Test configs reference checkpoints via a `weights:` template containing `{split}` and shell globs, e.g. `Summaries/.../summe_split{split}/best_tau_model/epoch*.ckpt`.

## Things to know before editing

- **Metrics**: Kendall τ and Spearman ρ come from `utils/evaluation_metrics.evaluate_summary`; F1 from `utils/visualize.f1_per_video`; `utils/generate_summary.generate_summary` does knapsack keyshot selection (`utils/knapsack_implementation.py`).
- **The aggregator is intentionally called with `mask=None`** in the multimodal pipeline (the masked variant is commented out in several places). Don't "fix" this without checking the surrounding comments.
- **Dim constants** live at the top of `projections.py` (`VISUAL_DIM=1024`, `AUDIO_DIM=512`, `TEXT_RAW_DIM=5120`, `FUSED_DIM=2048`, `COMP_DIM=256`); RL state dims derive from them in `rl_agent.py`. Change them there, not inline.
- **Actor init** uses tiny `final_std` so initial Dirichlet α≈1 (uniform modality prior); `alpha` is clamped `[0.1, 20]` to keep entropy finite. RL reward shaping is sensitive to the `*_coef` values — tune via YAML, not code.
- Some inline comments are in Turkish; preserve them when editing nearby code.
