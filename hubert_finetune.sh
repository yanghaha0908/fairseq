export CUDA_VISIBLE_DEVICES=7
export WANDB_NAME="finetune_checkpoint20"

/mnt/lustre/sjtu/home/xc915/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/finetune \
--config-name base_10h \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["ltr"] \
model.w2v_path=/mnt/lustre/sjtu/home/xc915/data/fairseq/None/checkpoints/checkpoint20.pt \
checkpoint.save_dir=/mnt/lustre/sjtu/home/xc915/data/fairseq/finecheck \
distributed_training.distributed_world_size=1 \
optimization.update_freq=[8] \
common.wandb_project="hubert" \


#checkpoint.restore_file=/mnt/lustre/sjtu/home/xc915/data/fairseq/None/checkpoints/checkpoint_best.pt
# 20 audios for test
