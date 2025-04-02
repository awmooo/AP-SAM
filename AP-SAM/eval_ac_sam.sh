torchrun  --nproc_per_node=2 train_AC-SAM.py \
--checkpoint ../pretrained_checkpoint/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AC-SAM/work_dirs/ap_sam_b_cnn_savebest_warm_768_lessfeat/vis \
--eval --restore-model /home/upc/Documents/SAM/sam-hq-main/sam-hq-main/train/AC-SAM/work_dirs/ap_sam_b_cnn_savebest_warm_768_lessfeat/best_model.pth \
#--visualize