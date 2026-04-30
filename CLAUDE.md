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

Key pinned versions: `torch==1.13.1+cu117`, `pytorch-lightning==1.5.10`. The PL version matters — API differences exist vs. newer versions (e.g., `validation_epoch_end` hook, `DDPPlugin`, `gpus=-1` trainer arg).

## Training & Evaluation

**Train all splits (SumMe)**:
```bash
bash train.sh
```

**Train single split**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
  --tag summe_split0 --model summe_head2_layer3 \
  --lr 0.000119 --epochs 200 --dataset summe \
  --reduced_dim 2048 --num_heads 2 --num_layers 3 \
  --split_idx 0 --pt_path 'llama_emb/summe_sum/'
```

**Evaluate all splits** (reads checkpoints from `Summaries/` and averages):
```bash
bash test.sh
```

**Evaluate single split** (direct):
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset summe --split_idx 0 --tag summe_split0 \
  --weights 'Summaries/summe_head2_layer3/summe/summe_split0/best_rho_model/<ckpt>.ckpt' \
  --pt_path llama_emb/summe_sum/ \
  --result_dir 'Summaries/summe_head2_layer3/summe/' \
  --num_heads 2 --num_layers 3 --reduced_dim 2048
```

Hyperparameters differ by dataset: SumMe `lr=0.000119`, TVSum `lr=0.00007`. 5-fold cross-validation (`split_idx` 0–4).

## Architecture

The model (`networks/model.py::LLMVS`) is a PyTorch Lightning module:

1. **Input**: Two LLaMA embeddings per frame (5120-dim each) — `user_prompt` (question/context) and `gen` (caption generation). Concatenated along the token dimension → shape `[frames, 2*tokens, 5120]`.
2. **Channel pooling** (`c_max_pooling`): `AdaptiveMaxPool1d(1)` collapses the token dimension → `[frames, 5120]`.
3. **Projection** (`d_linear1` + `LayerNorm`): linear to `reduced_dim` (default 2048).
4. **Global attention** (`transformer_encoder_agg`): `num_layers` Transformer encoder layers with `num_heads` heads — models inter-frame context.
5. **MLP head**: 5-layer MLP with `reduced_dim → 1`, sigmoid output → frame importance scores in [0,1].
6. **Summary generation** (`utils/generate_summary.py`): scores → segment scores → knapsack selection at 15% budget → binary summary vector.

Loss: MSE against ground-truth frame importance scores (`gtscore`). Metrics: Kendall's τ (`val_kTau`) and Spearman's ρ (`val_sRho`) against user annotations.

## Data Flow

- **Visual features**: `SumMe/eccv16_dataset_summe_google_pool5.h5` / `TVSum/eccv16_dataset_tvsum_google_pool5.h5` (Google Pool5 CNN features, not used in model forward pass but stored in dataset).
- **LLaMA embeddings**: `llama_emb/{dataset}_sum/user_prompt/user_prompt_pool.h5` and `llama_emb/{dataset}_sum/gen/gen_pool.h5` — max-pooled LLaMA-3 embeddings keyed by video filename.
- **Splits**: `dataset/summe_splits.json` and `dataset/tvsum_splits.json` — 5-fold train/test key lists.
- **Checkpoints**: saved to `Summaries/{model}/{dataset}/{tag}/best_rho_model/` and `best_tau_model/`, with `configuration.txt` logging hyperparameters.

## TVSum Evaluation Quirk

TVSum uses raw annotation TSV (`TVSum/ydata-tvsum50-anno.tsv`) at eval time — `evaluate_summary` parses it directly to compute per-annotator correlations. Video indices are 1-based, parsed from `video_name` suffix (e.g. `video_1`). SumMe uses mean user summary from the H5 file.
