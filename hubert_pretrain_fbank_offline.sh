export CUDA_VISIBLE_DEVICES=3
export WANDB_NAME="pretrain_fbank"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["km"] \
model.label_rate=50 \
checkpoint.save_interval=40 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/pretrain_fbank_ck \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
model._name=hubert \
# common.wandb_project="hubert" 

#nohup bash hubert_pretrain_fbank.sh >hubert_pretrain_fbank.log &
