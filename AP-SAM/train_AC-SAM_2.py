# modeified by HQ-SAM

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from segment_anything_single_decode_cnn.build_sam_ac import sam_model_registry

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize
from utils.loss_mask import loss_masks,sigmoid_ce_loss_jit,dice_loss_jit
from utils  import  misc
from sam_lora_image_encoder_qkv import LoRA_Sam

import matplotlib.pyplot as plt
from auto_prompt import Auto_mask_prompt

local_rank = int(os.environ["LOCAL_RANK"])
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'






def get_args_parser():
    parser = argparse.ArgumentParser('unet', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")


    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=20, type=int)
    parser.add_argument('--max_epoch_num', default=200, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
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


    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):
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
    if not args.eval:


        print("--- create training dataloader ---")
        train_im_gt_list,_= get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                            Resize(args.input_size)

                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

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

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True,output_device=args.local_rank)
    net_without_ddp = filter(lambda p: p.requires_grad, net.parameters())



 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch,gamma=0.5)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,valid_counts=valid_counts)
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

        evaluate(args, net, valid_dataloaders, args.visualize, valid_counts=valid_counts,prompt_signal=False)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,valid_counts=None):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        with open(f"{args.output}/info.txt", 'w') as info:
            info.write(str(args))
            print(f'save info.txt')

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)


    net.train()



    # Total loss
    epochs_train_loss = []
    # Dice loss
    epochs_loss_dice = []
    # BCE loss
    epochs_loss_mask = []

    epochs_prompt_loss_mask = []
    epochs_magnitude_loss = []

    epochs_eval_iou_biou = dict()


    for epoch in range(epoch_start,epoch_num):

        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter(name='training_loss',
                                meter=misc.SmoothedValue(fmt="batch_avg={value:.4f} (global_avg= {global_avg:.4f})"))
        metric_logger.add_meter(name='loss_dice',
                                meter=misc.SmoothedValue(fmt="batch_avg={value:.4f} (global_avg= {global_avg:.4f})"))
        metric_logger.add_meter(name='loss_mask',
                                meter=misc.SmoothedValue(fmt="batch_avg={value:.4f} (global_avg= {global_avg:.4f})"))
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,print_freq=1000):
            inputs, labels= data['image'], data['label']
            # print('image_path',train_dataset[data['imidx'][0]])
            # print('image_path',train_dataset[data['imidx'][1]])
            # mask_prompt = Auto_mask_prompt(inputs)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                # mask_prompt = mask_prompt.cuda()

            # h, w = inputs.shape[-2:]
            # padh = args.input_size[0] - h
            # padw = args.input_size[1] - w
            # labels = F.pad(labels, (0, padw, 0, padh))



            # mask_prompt = mask_prompt.type_as(inputs)
            # labels_256 = F.interpolate(mask_prompt, size=(256, 256), mode='bilinear')

            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            # labels_256 = F.pad(labels_256, (0, 0, 0, 36))
            # todo
            # labels_64 = F.avg_pool2d(labels,kernel_size=16,stride=16)
            # labels_256 = torch.softmax(labels_256.flatten(2),dim=-1).view(labels.shape[0],-1,64,64)
            trans_epoch = 100
            prompt_signal =(True if epoch-trans_epoch < 0 else False)




            batched_input = []
            for b_i in range(len(inputs)):
                dict_input = dict()


                dict_input['image'] = inputs[b_i]
                # dict_input['mask_inputs'] = labels_256[b_i:b_i + 1]
                dict_input['mask_inputs'] = labels_256[b_i:b_i + 1]
                dict_input['original_size'] = inputs[b_i].shape[-2:]

                batched_input.append(dict_input)

            output = net(batched_input, prompt_signal=prompt_signal)
            masks = [ item['origin_masks'] for item in output ]
            masks = torch.concat(masks,dim=0)


            # norm_prompt = F.interpolate(norm_prompt, size=(256, 256), mode='bilinear')



            # labels_lowres = labels_64.flatten(1)
            # max_value,_ = torch.max(labels_lowres,dim=-1,keepdim=True)
            # labels_lowres = labels_lowres/max_value
            # labels_lowres = torch.where(labels_lowres == 0, max_value*2, labels_lowres)
            # labels_lowres = torch.softmax(labels_lowres,dim=-1)

            loss_mask, loss_dice = loss_masks(masks, labels/255.0 , len(masks))
            norm_prompt = [item['norm_prompt'] for item in output]
            norm_prompt = torch.concat(norm_prompt, dim=0)
            norm_prompt_magnitude = norm_prompt
            # norm_prompt = torch.log_softmax(norm_prompt.flatten(2), dim=-1)

            gt_guide = [item['gt_guide'] for item in output]
            gt_guide = torch.concat(gt_guide, dim=0)
            gt_guide_magnitude = gt_guide
            # gt_guide = torch.softmax(gt_guide.flatten(2), dim=-1)



            # prompt_loss_mask = torch.nn.KLDivLoss(reduction='batchmean')(norm_prompt, gt_guide)
            magnitude_loss = torch.nn.MSELoss()(norm_prompt_magnitude,gt_guide_magnitude)
            # prompt_loss_mask = dice_loss_jit(norm_prompt,labels_lowres,len(norm_prompt))
            if prompt_signal:
                loss = loss_mask + loss_dice  + magnitude_loss
            else:
                loss_mask = loss_mask * (epoch-trans_epoch)/(epoch_num-trans_epoch)
                loss_dice = loss_dice * (epoch-trans_epoch)/(epoch_num-trans_epoch)

                loss = loss_mask + loss_dice  + magnitude_loss

            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice,
                         'magnitude_loss':magnitude_loss}

            # loss_dict = {"loss_dice": loss}
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        print('ps:in Averaged stats "batch_avg" is the last batch avg,"global_avg" is this epoch avg')
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}


        epochs_train_loss.append(train_stats['training_loss'])
        epochs_loss_mask.append(train_stats['loss_mask'])
        epochs_loss_dice.append(train_stats['loss_dice'])
        # epochs_prompt_loss_mask.append(train_stats['prompt_loss_mask'])
        epochs_magnitude_loss.append(train_stats['magnitude_loss'])

        lr_scheduler.step()
        test_stats = evaluate(args, net, valid_dataloaders,valid_counts=valid_counts,prompt_signal=prompt_signal)




        for key, value in test_stats.items():
            if key not in epochs_eval_iou_biou:
                epochs_eval_iou_biou[key] = []  # 如果键不在结果字典中，先初始化为一个空列表
            epochs_eval_iou_biou[key].append(value)  # 将当前值添加到结果字典的列表中




        train_stats.update(test_stats)




        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # save unet
    if misc.is_main_process():
        state_dict = net.state_dict()
        torch.save(state_dict, str(args.output + '/ac_sam{}.pth'.format(epoch)))
        print(f'Checkpoint {epoch} saved!')
    plot_loss_map(epochs_train_loss, epochs_loss_mask, epochs_loss_dice,epochs_prompt_loss_mask,epochs_magnitude_loss,outputdir=args.output)
    plot_eval_iou(epochs_eval_iou_biou, outputdir=args.output)

    with open(f"{args.output}/epoch{epoch_num}train_loss.txt", 'w') as train_loss:
        train_loss.write(str(epochs_train_loss))
        print(f'save{epoch_num}train_loss.txt')
    with open(f"{args.output}/epoch{epoch_num}loss_mask.txt", 'w') as loss_mask:
        loss_mask.write(str(epochs_loss_mask))
        print(f'save{epoch_num}loss_mask.txt')
    with open(f"{args.output}/epoch{epoch_num}loss_dice.txt", 'w') as loss_dice:
        loss_dice.write(str(epochs_loss_dice))
        print(f'save{epoch_num}loss_dice.txt')
    # with open(f"{args.output}/epoch{epoch_num}prompt_loss_mask.txt", 'w') as p_loss_dice:
    #     p_loss_dice.write(str(epochs_prompt_loss_mask))
    #     print(f'save{epoch_num}epochs_prompt_loss_mask.txt')
    with open(f"{args.output}/epoch{epoch_num}magnitude_loss.txt", 'w') as p_loss_dice:
        p_loss_dice.write(str(epochs_magnitude_loss))
        print(f'save{epoch_num}epochs_magnitude_loss.txt')
    with open(f"{args.output}/epoch{epoch_num}val_iou_biou.txt", 'w') as val_iou_biou:
        val_iou_biou.write(str(epochs_eval_iou_biou))
        print(f'save{epoch_num}val_iou_biou.txt')



def plot_loss_map(train_loss, loss_mask, loss_dice,prompt_loss_mask,magnitude_loss, outputdir):
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
    # plt.plot(range(len(prompt_loss_mask)), prompt_loss_mask, linewidth=1, linestyle="solid", label="prompt_loss_dice")
    plt.plot(range(len(magnitude_loss)), magnitude_loss, linewidth=1, linestyle="solid", label="magnitude_loss")
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

def evaluate(args, net, valid_dataloaders, visualize=False,valid_counts=None,prompt_signal=False):


    net.eval()
    print("Validating...")
    test_stats = {}

    total_iou = 0.0
    total_biou = 0.0

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori= data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            # mask_prompt = Auto_mask_prompt(inputs_val)





            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()
                # mask_prompt = mask_prompt.cuda()

            # mask_prompt = mask_prompt.type_as(inputs_val)
            # labels_256 = F.interpolate(mask_prompt, size=(256, 256), mode='bilinear')
            # todo
            # labels_256 = F.interpolate(labels_val, size=(256, 256), mode='bilinear')

            labels_256 = F.interpolate(labels_val, size=(256, 256), mode='bilinear')
            # labels_256 = F.pad(labels_256, (0, 0, 0, 36))


            batched_input = []
            for b_i in range(len(inputs_val)):
                dict_input = dict()

                dict_input['image'] = inputs_val[b_i]
                if prompt_signal:
                    dict_input['mask_inputs'] = labels_256[b_i:b_i + 1]
                else:
                    dict_input['mask_inputs'] = None
                dict_input['original_size'] = inputs_val[b_i].shape[-2:]
                batched_input.append(dict_input)
            with torch.no_grad():
                output = net(batched_input, prompt_signal=prompt_signal)
            masks = [item['origin_masks'] for item in output]
            masks = torch.concat(masks, dim=0)





            iou = compute_iou(masks,labels_ori)
            boundary_iou = compute_boundary_iou(masks,labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks.detach(), args.input_size, mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, None, None, save_base , imgs_ii, show_iou, show_boundary_iou)
                       

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        if valid_counts is not None:
            weight = valid_counts[k]/ valid_counts[-1]
            total_iou += weight * resstat[f'val_iou_{k}']
            total_biou += weight * resstat[f'val_boundary_iou_{k}']

        test_stats.update(resstat)
        test_stats.update({'total_iou':total_iou})
        test_stats.update({'total_biou': total_biou})
    if valid_counts is not None:
        print(f"total_iou={total_iou:.4f},total_biou={total_biou:.4f}")

    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

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



    # valid set

    dataset_crack0_5_val = {"name": "Crack",
                        "im_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val/images",
                        "gt_dir": "/home/upc/Documents/Cracksegdata/split_data/0-5um/val/masks",
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



    train_datasets = [dataset_crack0_5]
    valid_datasets = [dataset_crack0_5_val]

    args = get_args_parser()

    ac_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    #  lora rank = 4
    net = LoRA_Sam(ac_sam, 4)

    main(net, train_datasets, valid_datasets, args)
