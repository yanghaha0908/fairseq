export CUDA_VISIBLE_DEVICES=6
export WANDB_NAME="finetune_checkpoint20_chunwav"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/finetune \
--config-name base_10h \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["ltr"] \
dataset.train_subset=train_clean_100 \
dataset.valid_subset=dev_other \
dataset.num_workers=6 \
dataset.skip_invalid_size_inputs_valid_test=true \
model.w2v_path=/mnt/lustre/sjtu/home/gry10/fairseq/None/checkpoints/checkpoint20.pt \
model.mask_prob=0.65 \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/finecheck/fine20_chunwav \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
optimization.lr=[0.00003] \
optimization.max_update=80000 \
lr_scheduler.warmup_steps=8000 \
lr_scheduler.hold_steps=32000 \
lr_scheduler.decay_steps=40000 \
common.wandb_project="hubert" \




# wav2vec
