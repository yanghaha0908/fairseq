export CUDA_VISIBLE_DEVICES=0,1

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/small20 \
task.label_dir=/data/ygr/small20 \
task.labels=["km"] \
model.label_rate=50 \
distributed_training.distributed_world_size=2 \
optimization.update_freq=[8] \
checkpoint.save_interval=40 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/data/ygr/small20/che \
optimization.max_update=10 \




# wav2vec change max_update=50000
#nohup bash hubert_finetune_official.sh >finetune_official.log &
