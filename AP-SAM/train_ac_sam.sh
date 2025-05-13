torchrun --nproc_per_node=2 train_AC-SAM.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output work_dirs/ac_sam_b_mudstone_fullfineencoder \
--warmup