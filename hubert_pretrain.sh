export CUDA_VISIBLE_DEVICES=2
export WANDB_NAME="run_again"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["km"] \
model.label_rate=50 \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/pretrain_origin \
checkpoint.save_interval=10 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
# common.wandb_project="hubert" \
