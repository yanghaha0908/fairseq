export CUDA_VISIBLE_DEVICES=0,1,2,5

/mnt/lustre/sjtu/home/xc915/anaconda3/envs/fairseq/bin/python fairseq_cli/hydra_train.py \
--config-dir examples/hubert/config/pretrain\
--config-name hubert_base_librispeech \
task.data=/data/ygr/shu \
task.label_dir=/data/ygr/k500 \
task.labels=["km"] \
model.label_rate=50 \
common.wandb_project=hubert \
optimization.max_update=10

