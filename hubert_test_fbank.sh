export CUDA_VISIBLE_DEVICES=4,5
export WANDB_NAME="pretrain_fbank"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/small20 \
task.label_dir=/data/ygr/small20 \
task.labels=["km"] \
model.label_rate=50 \
distributed_training.distributed_world_size=2 \
optimization.update_freq=[4] \
checkpoint.save_interval=4 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/data/ygr/small20/che \
model._name=hubert_fbank \
common.wandb_project="hubert" 

#nohup bash hubert_pretrain_fbank.sh >hubert_pretrain_fbank.log &
