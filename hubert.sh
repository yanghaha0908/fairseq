export CUDA_VISIBLE_DEVICES=0,1,2,5
export WANDB_NAME="run_again"

/mnt/lustre/sjtu/home/xc915/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["km"] \
model.label_rate=50 \
distributed_training.distributed_world_size=4 \
common.wandb_project="hubert" \
checkpoint.save_interval=10 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
optimization.update_freq=[8]
