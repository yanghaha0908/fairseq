export CUDA_VISIBLE_DEVICES=2

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/finetune \
--config-name base_10h \
task.data=/data/ygr/small20 \
task.label_dir=/data/ygr/small20 \
task.labels=["ltr"] \
dataset.num_workers=6 \
dataset.skip_invalid_size_inputs_valid_test=true \
model.w2v_path=/mnt/lustre/sjtu/home/gry10/fairseq/finetune/checkpoint/fineofficial/checkpoint_best.pt \
model.mask_prob=0.65 \
dataset.validate_after_updates=0 \
dataset.validate_interval=1 \
checkpoint.save_dir=/data/ygr/small20/chel \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
optimization.lr=[0.00003] \
optimization.max_update=50000 \
lr_scheduler.warmup_steps=8000 \
lr_scheduler.hold_steps=32000 \
lr_scheduler.decay_steps=40000 \




# wav2vec change max_update=50000
#nohup bash hubert_finetune_official.sh >finetune_official.log &
