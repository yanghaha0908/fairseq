export CUDA_VISIBLE_DEVICES=0,1


/mnt/lustre/sjtu/home/xc915/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain \
--config-name hubert_base_librispeech \
task.data=/data/ygr/small20 \
task.label_dir=/data/ygr/small20 \
task.labels=["km"] \
model.label_rate=50 \
distributed_training.distributed_world_size=2 \
optimization.max_update=8 \
checkpoint.save_interval=4 \
checkpoint.save_interval_updates=2 \
checkpoint.keep_interval_updates=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_dir=/data/ygr/small20/che \
dataset.validate_interval=1


#checkpoint.restore_file=/mnt/lustre/sjtu/home/xc915/data/fairseq/None/checkpoints/checkpoint_best.pt
# 20 audios for test
