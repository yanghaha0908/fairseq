export CUDA_VISIBLE_DEVICES=1

/mnt/lustre/sjtu/home/gry10/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/finetune \
--config-name base_10h \
task.data=/data/ygr/small20 \
task.label_dir=/data/ygr/small20 \
task.labels=["ltr"] \
model.w2v_path=/mnt/lustre/sjtu/home/xc915/data/fairseq/None/checkpoints/checkpoint20.pt \
checkpoint.save_dir=/data/ygr/small20/f \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
dataset.validate_interval=1 \
optimization.max_update=20 



#checkpoint.restore_file=/mnt/lustre/sjtu/home/xc915/data/fairseq/None/checkpoints/checkpoint_best.pt
# 20 audios for test
