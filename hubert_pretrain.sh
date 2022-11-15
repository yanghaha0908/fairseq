export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_NAME="run_again"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["km"] \
model.label_rate=50 \
distributed_training.distributed_world_size=2 \
optimization.update_freq=[8] \
checkpoint.save_interval=10 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/pretrain_origin_testtime \

# common.wandb_project="hubert" \
