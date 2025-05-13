torchrun  --nproc_per_node=2 eval_APSAM.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AP-SAM/vis \
--eval --restore-model /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AP-SAM/best_model.pth\

#--visualize