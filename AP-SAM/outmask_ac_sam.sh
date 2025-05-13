torchrun  --nproc_per_node=2 output_binary_masks.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AP-SAM/work_dirs/ap_sam_b_mudstone_r4/vis \
--eval --restore-model /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AP-SAM/work_dirs/ap_sam_b_mudstone_r4/best_model.pth \
#--visualize