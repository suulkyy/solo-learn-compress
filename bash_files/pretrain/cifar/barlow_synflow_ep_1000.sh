python3 ../../../main_pretrain.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --data_dir ~/workspace/datasets/ \
    --max_epochs 1000 \
    --gpus 7 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --pruner synflow \
    --sparsity 0.001 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name barlow_res18_synflow_pruning_0.001 \
    --project CIFAR100-barlow-1000ep \
    --entity kaistaim2 \
    --wandb \
    --save_checkpoint \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --scale_loss 0.1 \
    --knn_eval \
