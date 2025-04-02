# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import re

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder,MaskDecoder2_ab,MaskDecoder_ab, Sam, Sam_Twostage,Sam_Twostage_ablation, DeformableTwoWayTransformer,TwoWayTransformer,MaskDecoder2,LightedTwoWayTransformer


def build_sam_vit_h(checkpoint=None,):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry_ab = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,

):
    prompt_embed_dim = 256
    image_size = [880,1024]
    vit_patch_size = 16
    image_embedding_size = [image_size[0] // vit_patch_size,image_size[1] // vit_patch_size]
    sam = Sam_Twostage_ablation(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size[0], image_embedding_size[1]),
            input_image_size=(image_size[0], image_size[1]),
            mask_in_chans=16,

            ),
        mask_decoder=MaskDecoder_ab(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            # img_size=image_size,

        ),
        mask_decoder2=MaskDecoder2_ab(
            # img_size=image_size,
            mask_in_chans=16,
            transformer=TwoWayTransformer(
                depth=1,
                embedding_dim = prompt_embed_dim//2,
                mlp_dim=2048,
                num_heads=8,

            ),
            transformer2=TwoWayTransformer(
                depth=1,
                embedding_dim= prompt_embed_dim//2,
                mlp_dim=2048,
                num_heads=8,

            ),
            # transformer3=TwoWayTransformer(
            #     depth=1,
            #     embedding_dim=prompt_embed_dim // 8,
            #     mlp_dim=2048,
            #     num_heads=8,
            #
            # ),
            transformer_dim=prompt_embed_dim//8,
            img_size=image_size,
        ),
        pixel_mean=[137.815,137.815,137.815],
        pixel_std=[37.201,37.201,37.201],
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
    )

    # sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if image_size == [1024,1024]:
            sam.load_state_dict(state_dict,strict=False)
        else:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size,encoder_global_attn_indexes)
            sam.load_state_dict(new_state_dict)
    return sam

# modified from SAMed,it modifies the positional encoding to adapt the new img_size(like 512x512),after the function it masks SAM adapt diffient resolution
def load_from(sam, state_dict, image_size, vit_patch_size,encoder_global_attn_indexes):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = [image_size[0] // vit_patch_size, image_size[1]//vit_patch_size]
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size[0], token_size[1]), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_h_keys = []
        global_rel_pos_w_keys = []
        for k in rel_pos_keys:
            split_list = k.split('.')
            index = split_list[2]
            for i in encoder_global_attn_indexes:
                if  int(index) == i:
                    if 'rel_pos_h' in k:
                        global_rel_pos_h_keys.append(k)
                    else:
                        global_rel_pos_w_keys.append(k)

        for k,j in zip(global_rel_pos_h_keys,global_rel_pos_w_keys):
            rel_pos_h_params = new_state_dict[k]
            rel_pos_w_params = new_state_dict[j]
            _, hidden_dim = rel_pos_h_params.shape
            rel_pos_h_params = rel_pos_h_params.unsqueeze(0).unsqueeze(0)
            rel_pos_h_params = F.interpolate(rel_pos_h_params, (token_size[0] * 2 - 1, hidden_dim), mode='bilinear', align_corners=False)
            rel_pos_w_params = rel_pos_w_params.unsqueeze(0).unsqueeze(0)
            rel_pos_w_params = F.interpolate(rel_pos_w_params, (token_size[1] * 2 - 1, hidden_dim), mode='bilinear',align_corners=False)
            new_state_dict[k] = rel_pos_h_params[0, 0, ...]
            new_state_dict[j] = rel_pos_w_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
