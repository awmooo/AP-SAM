# modified from SAMed https://github.com/hitachinsk/SAMed/blob/main/sam_lora_image_encoder.py

import re


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
# from segment_anything_ap.modeling import Sam

from icecream import ic


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,

            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv

        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, self.dim:2*self.dim] += new_k
        qkv[:, :, :, -self.dim:] += new_v

        return qkv


class _LoRA_o(nn.Module):
    """
    """

    def __init__(
            self,
            proj: nn.Module,
            linear_a_o: nn.Module,
            linear_b_o: nn.Module,

    ):
        super().__init__()
        self.proj = proj
        self.linear_a_o = linear_a_o
        self.linear_b_o = linear_b_o


    def forward(self, x):
        proj = self.proj(x) # B,N,N,C
        new_o = self.linear_b_o(self.linear_a_o(x))
        proj += new_o


        return proj


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        prompt_encoder_no_train = ['point_embeddings','not_a_point_embed','mask_downscaling']
        mask_decoder_no_train = ['iou_prediction_head','output_hypernetworks_mlps',]

        # image_encoder_need_train = ['post_pos_embed','conv_down']



        # lets freeze first  freeze image_encoder , some of prompt_encoder and mask_decoder
        for name,param in sam_model.image_encoder.named_parameters():
            # if 'patch_embed' in name :
            #         # or 'rel_pos_h'in name or 'rel_pos_w' in name:
            #     param.requires_grad = True
            # else:
            param.requires_grad = False
            # for item in image_encoder_need_train:
            #     if re.match(item, name):
            #         param.requires_grad = True
            #         break

            # param.requires_grad = False
        # do not fine-tuning param
        for name,param in sam_model.prompt_encoder.named_parameters():
            for item in prompt_encoder_no_train:
                if  re.match(item,name):
                    param.requires_grad = False
                    break

        # do not fine-tuning param
        for name,param in sam_model.mask_decoder.named_parameters():
            for item in mask_decoder_no_train:
                if re.match(item, name):
                    param.requires_grad = False
                    break


        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            # todo projection
            w_proj = blk.attn.proj

            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            w_a_linear_o = nn.Linear(self.dim, r, bias=False)
            w_b_linear_o = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_As.append(w_a_linear_o)
            self.w_Bs.append(w_b_linear_o)
            blk.attn.proj = _LoRA_o(w_proj,
                                    w_a_linear_o,
                                    w_b_linear_o)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        prompt_encoder2_tensors = {}
        mask_decoder_tensors = {}
        mask_decoder2_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'prompt_encoder2' in key:
                prompt_encoder2_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'mask_decoder2' in key:
                mask_decoder2_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors,**prompt_encoder2_tensors, **mask_decoder_tensors, **mask_decoder2_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load prompt encoder
        prompt_encoder2_keys = [k for k in sam_keys if 'prompt_encoder2' in k]
        prompt_encoder2_values = [state_dict[k] for k in prompt_encoder2_keys]
        prompt_encoder2_new_state_dict = {k: v for k, v in zip(prompt_encoder2_keys, prompt_encoder2_values)}
        sam_dict.update(prompt_encoder2_new_state_dict)

        # load mask decoder
        mask_decoder2_keys = [k for k in sam_keys if 'mask_decoder2' in k]
        mask_decoder2_values = [state_dict[k] for k in mask_decoder2_keys]
        mask_decoder2_new_state_dict = {k: v for k, v in zip(mask_decoder2_keys, mask_decoder2_values)}
        sam_dict.update(mask_decoder2_new_state_dict)

        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input,prompt_signal):
        return self.sam(batched_input,prompt_signal)



