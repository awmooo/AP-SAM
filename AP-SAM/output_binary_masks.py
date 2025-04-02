# modeified by HQ-SAM

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from segment_anything_ap.build_sam_ac import sam_model_registry

from utils.dataloader import get_im_gt_name_dict, create_dataloaders,Tonorm, RandomHFlip, Resize,RandomVFlip,RandomBrightnessContrast,ContrastEnhancement
from utils.loss_mask import loss_masks,sigmoid_ce_loss_jit,dice_loss_jit
from utils  import  misc
from sam_lora_image_encoder_qkv import LoRA_Sam

import matplotlib.pyplot as plt
from auto_prompt import Auto_mask_prompt
from PIL import Image

local_rank = int(os.environ["LOCAL_RANK"])
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(masks, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())


        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()




def get_args_parser():
    parser = argparse.ArgumentParser('unet', add_help=False)

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")


    parser.add_argument("--model-type", type=str, default="vit_b",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    # parser.add_argument('--lr_drop_epoch', default=75, type=int)
    parser.add_argument('--max_epoch_num', default=200, type=int)
    parser.add_argument('--input_size', default=[880,1024], type=list)
    parser.add_argument('--batch_size_train', default=2, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=50, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument('--warmup', action='store_true',
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=100,
                        help='Warp up iterations, only valid when warmup is activated')


    return parser.parse_args()


def main(net, valid_datasets, args):
    # include cuda.set_device() and distributed.init_process_group()
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    ### --- Step 1: Train or Valid dataset ---



    print("--- create valid dataloader ---")
    valid_im_gt_list,valid_counts = get_im_gt_name_dict(valid_datasets, flag="valid")
    print(f'Total_counts:{valid_counts}')
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                              Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()


    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print(name)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=False,output_device=args.local_rank)
    net_without_ddp = filter(lambda p: p.requires_grad, net.parameters())




    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
       pass
    else:

        if args.restore_model:
            net_without_ddp = net
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                state_dict = torch.load(args.restore_model)

                new_state_dict = {}

                for key in state_dict.keys():
                    new_key = 'module.'+ key
                    new_state_dict[new_key] = state_dict[key]
                net_without_ddp.load_state_dict(new_state_dict)
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        evaluate(args, net, valid_dataloaders, args.visualize, valid_counts=valid_counts)






def plot_loss_map(train_loss, loss_mask, loss_dice, outputdir):
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(range(len(train_loss)), train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(range(len(loss_mask)), loss_mask, linewidth=1, linestyle="solid", label="loss_mask(bce)")
    plt.plot(range(len(loss_dice)), loss_dice, linewidth=1, linestyle="solid", label="loss_dice")
    # plt.plot(range(len(loss_mask_low)), loss_mask_low, linewidth=1, linestyle="solid", label="loss_mask_low")
    # plt.plot(range(len(loss_dice_low)), loss_dice_low, linewidth=1, linestyle="solid", label="loss_dice_low")

    # plt.plot(range(len(prompt_loss_mask)), prompt_loss_mask, linewidth=1, linestyle="solid", label="prompt_loss_dice")
    plt.legend()
    plt.title('Loss curve')

    plt.savefig(f"{outputdir}/epoch{len(train_loss)}train_loss.svg", dpi=300, format='svg')
    plt.close()


def plot_eval_iou(eval_iou_dict, outputdir):
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('IoU')  # y轴标签

    epoch = -1
    for k, v in eval_iou_dict.items():
        plt.plot(range(len(v)), v, linewidth=1, linestyle="solid", label=f"{k}")
        epoch = len(v)

    plt.legend()
    plt.title('IoU curve')

    plt.savefig(f"{outputdir}/epoch{epoch}val_iou_biou.svg", dpi=300, format='svg')
    plt.close()



def compute_dice(preds, target):
    if not isinstance(preds, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("Both preds and target must be torch.Tensors.")
    assert target.shape[1] == 1, 'only support one mask per image now'
    if preds.shape[2:] != target.shape[2:]:
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    dice_sum = 0
    for i in range(len(preds)):
        dice_sum += misc.mask_dice(postprocess_preds[i], target[i])
    return dice_sum / len(preds)

def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i], target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i], postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, net, valid_dataloaders, visualize=False,valid_counts=None):


    net.eval()
    print("Validating...")
    test_stats = {}

    total_iou = 0.0
    total_biou = 0.0
    total_dice = 0.0

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            gt_path,imidx_val, inputs_val, labels_val, shapes_val, labels_ori= data_val['ori_gt_path'],data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            # mask_prompt = Auto_mask_prompt(inputs_val)





            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()
                # mask_prompt = mask_prompt.cuda()

            # mask_prompt = mask_prompt.type_as(inputs_val)
            # labels_256 = F.interpolate(mask_prompt, size=(256, 256), mode='bilinear')
            # todo
            # labels_256 = F.interpolate(labels_val, size=(256, 256), mode='bilinear')

            # labels_256 = F.interpolate(labels_val, size=(256, 256), mode='bilinear')
            # labels_256 = F.pad(labels_256, (0, 0, 0, 36))


            batched_input = []
            for b_i in range(len(inputs_val)):
                dict_input = dict()

                dict_input['image'] = inputs_val[b_i]
                gt_path_1 = os.path.basename(gt_path[b_i])


                # dict_input['mask_inputs'] = None
                dict_input['original_size'] = inputs_val[b_i].shape[-2:]
                batched_input.append(dict_input)
            with torch.no_grad():
                output = net(batched_input, prompt_signal=False)
            masks = [item['masks'] for item in output]
            masks = torch.concat(masks, dim=0)
            masks = masks.int()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()


            masks = masks.float().squeeze(0).squeeze(0)



            mask_image = Image.fromarray((masks.cpu().numpy() * 255).astype('uint8'))
            os.makedirs(args.output, exist_ok=True)
            save_name = args.output+f'/{gt_path_1}.jpg'
            mask_image.save(save_name,'JPEG',quality=100)
















    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_muskstone = {"name": "Crack",
                        "im_dir": "/home/upc/Documents/mudstone/split/train/images",
                        "gt_dir": "/home/upc/Documents/mudstone/split/train/labels",
                        "im_ext": ".jpg",
                        "gt_ext": ".jpg"}

    dataset_crack0_5 = {"name": "Crack",
                 "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/train/images",
                 "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/train/masks",
                 "im_ext": ".png",
                 "gt_ext": ".png"}

    dataset_crack10_50 = {"name": "Crack",
                     "im_dir": "/home/upc/Documents/Cracksegdata/split_data/10-50um/train/images",
                     "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/10-50um/train/masks",
                     "im_ext": ".png",
                     "gt_ext": ".png"}

    dataset_crack100_500 = {"name": "Crack",
                     "im_dir": "/home/upc/Documents/Cracksegdata/split_data/100-500um/train/images",
                     "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/100-500um/train/masks",
                     "im_ext": ".png",
                     "gt_ext": ".png"}

    dataset_crack1000 = {"name": "Crack",
                            "im_dir": "/home/upc/Documents/Cracksegdata/split_data/1000um/train/images",
                            "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/1000um/train/masks",
                            "im_ext": ".png",
                            "gt_ext": ".png"}

    dataset_crack0_5_fy = {"name": "Crack",
                        "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um-fy/train/images",
                        "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um-fy/train/masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}



    # valid set

    dataset_muskstone_val = {"name": "Crack",
                         "im_dir": "/home/upc/Documents/mudstone/split/val/images",
                         "gt_dir": "/home/upc/Documents/mudstone/split/val/labels",
                         "im_ext": ".jpg",
                         "gt_ext": ".jpg"}

    dataset_crack0_5_val = {"name": "Crack",
                        "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val/images",
                        "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val/masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_crack0_5_test = {"name": "Crack",
                            "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/test/images",
                            "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/test/masks",
                            "im_ext": ".png",
                            "gt_ext": ".png"}

    dataset_crack10_50_val = {"name": "Crack",
                          "im_dir": "/home/upc/Documents/Cracksegdata/split_data/10-50um/val/images",
                          "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/10-50um/val/masks",
                          "im_ext": ".png",
                          "gt_ext": ".png"}

    dataset_crack100_500_val = {"name": "Crack",
                            "im_dir": "/home/upc/Documents/Cracksegdata/split_data/100-500um/val/images",
                            "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/100-500um/val/masks",
                            "im_ext": ".png",
                            "gt_ext": ".png"}

    dataset_crack1000_val = {"name": "Crack",
                         "im_dir": "/home/upc/Documents/Cracksegdata/split_data/1000um/val/images",
                         "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/1000um/val/masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    dataset_crack0_5_val_filted = {"name": "Crack",
                            "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val_filted/images",
                            "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val_filted/masks",
                            "im_ext": ".png",
                            "gt_ext": ".png"}

    dataset_crack0_5_fy_val = {"name": "Crack",
                           "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um-fy/val/images",
                           "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um-fy/val/masks",
                           "im_ext": ".png",
                           "gt_ext": ".png"}



    valid_datasets = [dataset_muskstone_val]

    args = get_args_parser()

    ac_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    #  lora rank = 4
    ac_sam = LoRA_Sam(ac_sam, 4)

    # if not args.eval:
    #     mask_decoder = ac_sam.sam.mask_decoder
    #     mask_decoder.ac_pred_head.load_state_dict(mask_decoder.output_hypernetworks_mlps[0].state_dict())
    #     mask_decoder.output_upscaling_2.load_state_dict(mask_decoder.output_upscaling.state_dict())
    #     mask_decoder.transformer_cnn.load_state_dict(mask_decoder.transformer.state_dict())
    #     mask_decoder.cnn_tokens.load_state_dict(mask_decoder.mask_tokens.state_dict())




    # todo uncomment under code to full fine-tuning SAM image encoder and comment the 'net = LoRA_sam(...)'
    # import re
    # #
    # prompt_encoder_no_train = ['point_embeddings','not_a_point_embed','no_mask_embed','mask_downscaling']
    # mask_decoder_no_train = ['iou_prediction_head']
    #
    # for name,param in ac_sam.prompt_encoder.named_parameters():
    #     for item in prompt_encoder_no_train:
    #         if  re.match(item,name):
    #             param.requires_grad = False
    #             break
    #
    # # do not fine-tuning param
    # for name,param in ac_sam.mask_decoder.named_parameters():
    #     for item in mask_decoder_no_train:
    #         if re.match(item, name):
    #             param.requires_grad = False
    #             break
    # # todo
    # for n, value in ac_sam.image_encoder.named_parameters():
    #     if "Adapter" not in n:
    #         value.requires_grad = False
    main(ac_sam, valid_datasets, args)
