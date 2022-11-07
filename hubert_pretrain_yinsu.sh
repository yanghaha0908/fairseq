export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_NAME="pretrain_yinsu"

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/data2vec_jt \
task.label_dir=/data/ygr/data2vec_jt \
task.labels=["phn"] \
dataset.train_subset=train_960 \
dataset.valid_subset=val_100 \
model.label_rate=50 \
distributed_training.distributed_world_size=4 \
optimization.update_freq=[8] \
checkpoint.save_interval=10 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/mnt/lustre/sjtu/home/gry10/fairseq/pretrain_yinsu_ck \
common.wandb_project="hubert" \


#nohup bash hubert_pretrain_yinsu.sh >hubert_pretrain_yinsu.log &
