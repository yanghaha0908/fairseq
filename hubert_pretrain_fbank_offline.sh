export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_NAME="pretrain_fbank_offline"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/librispeech/npyfiles \
task.label_dir=/data/ygr/k500 \
dataset.valid_subset=dev_clean \
task.labels=["km"] \
model.label_rate=50 \
checkpoint.save_interval=5 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/pretrain_fbank_ck \
distributed_training.distributed_world_size=4 \
optimization.update_freq=[8] \
model._name=hubert_fbank_offline \
task._name=hubert_fbank_pretraining \
dataset.max_tokens=8750 \
task.sample_rate=100 \
common.wandb_project="hubert" \

#nohup bash hubert_pretrain_fbank_offline.sh >hubert_pretrain_fbank_offline.log &
