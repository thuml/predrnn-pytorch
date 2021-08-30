export CUDA_VISIBLE_DEVICES=1
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name bair \
    --train_data_paths /data/Action-BAIR/ \
    --valid_data_paths /data/Action-BAIR/ \
    --save_dir checkpoints/bair_action_cond_predrnn_v2 \
    --gen_frm_dir results/bair_action_cond_predrnn_v2 \
    --model_name action_cond_predrnn_v2 \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 2 \
    --total_length 12 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 1 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 16 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --conv_on_input 1 \
    --res_on_conv 1 \
#    --pretrained_model ./checkpoints/bair_predrnn_v2/bair_model.ckpt