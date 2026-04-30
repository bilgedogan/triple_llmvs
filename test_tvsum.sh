CUDA_VISIBLE_DEVICES=0 python test_splits.py --dataset tvsum --weights tau \
        --result_dir Summaries/tvsum_head2_layer3/ \
        --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048