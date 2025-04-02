# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .prompt_encoder import PositionEmbeddingRandom
from .unet_parts import DoubleConv
from .ASPP import ASPP

class MaskDecoder2(nn.Module):
    def __init__(
        self,
        *,
        mask_in_chans=16,
        transformer_dim: int,
        transformer: nn.Module,
        # transformer2: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        img_size=[880, 1024],

    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.
        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks

        """
        super().__init__()
        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(mask_in_chans // 4),
        #     activation(),
        #     nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
        #     LayerNorm2d(mask_in_chans),
        #     activation(),
        #     nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        # )

        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         1,
        #         transformer_dim,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(transformer_dim),
        #     nn.Conv2d(
        #         transformer_dim,
        #         transformer_dim,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(transformer_dim),
        # )

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        # self.transformer2 = transformer2

        # self.ASPP = nn.Sequential(
        #     LayerNorm2d(transformer_dim * 4),
        #     ASPP(transformer_dim * 4, transformer_dim),
        # )



        # todo ac_token for crack seg masks
        # self.ac_tokens = nn.Embedding(1, transformer_dim)
        #
        # self.cnn_tokens = nn.Embedding(1, transformer_dim)

        # self.aspp_tokens = nn.Embedding(1, transformer_dim)
        self.pe_layer = PositionEmbeddingRandom(transformer_dim *4//2)

        # self.level_embed = nn.Parameter(torch.Tensor(2, transformer_dim * 4))

        self.posemb_1 = self.pe_layer([img_size[0]//4,img_size[1]//4]).unsqueeze(0)

        self.pe_layer_2 = PositionEmbeddingRandom(transformer_dim * 2 // 2)

        self.posemb_2 = self.pe_layer_2([img_size[0]//2,img_size[1]//2]).unsqueeze(0)

        self.upblock = Upblock()



        #
        # # self.early_upscaling = nn.Sequential(
        # #     nn.ConvTranspose2d(256,  transformer_dim*2, kernel_size=2, stride=2),
        # #     LayerNorm2d(transformer_dim*2),
        # #     activation(),
        # #     nn.ConvTranspose2d(transformer_dim*2 , transformer_dim, kernel_size=2, stride=2),
        # #     LayerNorm2d(transformer_dim ),
        # #     activation(),
        # #
        # # )
        #
        # self.cnn_downdim = nn.Sequential(
        #     nn.Conv2d(256, transformer_dim *4 , kernel_size=1, stride=1,bias=True),
        #     # LayerNorm2d(transformer_dim ),
        #     # activation(),
        #     # nn.Conv2d(transformer_dim , transformer_dim,
        #     #           kernel_size=3, stride=1,padding=1,bias=True),
        #     # LayerNorm2d(transformer_dim),
        #     # activation(),
        #
        # )
        #
        # self.imgconv = nn.Sequential(
        #
        #     nn.Conv2d(transformer_dim, transformer_dim*2,
        #               kernel_size=1, stride=1,bias=True),
        #     # LayerNorm2d(transformer_dim ),
        #     # activation(),
        #
        # )
        #
        # self.cnn_transf = nn.Sequential(
        #     nn.Conv2d(transformer_dim, transformer_dim*2,
        #               kernel_size=1, stride=1,bias=True),
        #     # LayerNorm2d(transformer_dim),
        #     # activation(),
        #
        # )
        #
        # # self.third_cnn_transf = nn.Sequential(
        # #     nn.Conv2d(512, transformer_dim *4, kernel_size=1, stride=1, bias=True),
        # #     LayerNorm2d(transformer_dim *4),
        # #     activation(),
        # #     nn.ConvTranspose2d(transformer_dim *4, transformer_dim , kernel_size=2, stride=2),
        # #     LayerNorm2d(transformer_dim ),
        # #     activation(),
        # #     nn.Conv2d(transformer_dim , transformer_dim, kernel_size=3, stride=1,padding=1,bias=True),
        # #     LayerNorm2d(transformer_dim),
        # #     activation(),
        # # )
        #
        # self.first_cnn_transf = nn.Sequential(
        #     nn.Conv2d(128, transformer_dim *4 ,
        #              kernel_size=1, stride=1, bias=True),
        #     # LayerNorm2d(transformer_dim ),
        #     # activation(),
        #     # nn.Conv2d(transformer_dim , transformer_dim ,
        #     #           kernel_size=3, stride=1, padding=1, bias=True),
        #     # LayerNorm2d(transformer_dim ),
        #     # activation(),
        # )
        #
        #
        #
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim *4, transformer_dim *4 , kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim *4),
        #     activation(),
        #     # nn.Conv2d(transformer_dim * 2, transformer_dim * 2,
        #     #           kernel_size=3, stride=1, padding=1, bias=True),
        #     # LayerNorm2d(transformer_dim * 2),
        # )

        # self.output_upscaling_2 =  nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim *4, transformer_dim *4, kernel_size=2, stride=2),
        #     activation(),
        #
        #
        # )

        self.feat_fusion = nn.Sequential(
            nn.Conv2d(transformer_dim * 2, transformer_dim *2,1, 1,
                      bias=True),
            LayerNorm2d(transformer_dim *2 ),
            activation(),
            nn.ConvTranspose2d(transformer_dim * 2, transformer_dim * 2, kernel_size=2, stride=2),
            activation(),
            # nn.Conv2d(transformer_dim * 2, transformer_dim * 5, 1, 1,
            #           bias=True),
            # LayerNorm2d(transformer_dim * 6),
            # activation(),


        )

        # self.feat_fusion_2 = nn.Sequential(
        #     nn.Conv2d(transformer_dim *  4, transformer_dim *4, 1, 1,
        #               bias=True),
        #     LayerNorm2d(transformer_dim *4),
        #
        # )




        # self.adjust_std = nn.Parameter(torch.tensor(1.0))







        # todo ac mask pred head
        self.ac_pred_head_sam = MLP(transformer_dim  *8, transformer_dim *8, transformer_dim *4, 3)
        # self.ac_pred_head_cnn = MLP(transformer_dim  , transformer_dim, transformer_dim *7, 3)


        # todo
        self.ac_pred_head_1 = MLP(transformer_dim  *2, transformer_dim*4, transformer_dim *2 , 3)
        # self.ac_pred_head_2 = MLP(transformer_dim , transformer_dim *2, transformer_dim , 3)




    def forward(
        self,
        image_embeddings: torch.Tensor,
        early_embeddings: torch.Tensor,
        ac_token : torch.Tensor,
        cnn_token: torch.Tensor,
        cnn_feature : torch.Tensor,
        cnn_first_feature: torch.Tensor,
        up_cnn_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder(b,32,256,256)
          image_pe (torch.Tensor): positional encoding with the shape of image_embedding
          one_stage_masks (torch.Tensor): one stage mask_decoder(from vanilla SAM)output masks


        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            early_embeddings =early_embeddings,
            ac_token = ac_token,
            cnn_token =cnn_token,
            cnn_feature = cnn_feature,
            cnn_first_feature = cnn_first_feature,
            up_cnn_feature = up_cnn_feature
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)


        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        early_embeddings:torch.Tensor,
        ac_token:torch.Tensor,
        cnn_token:torch.Tensor,
        cnn_feature:torch.Tensor,
        cnn_first_feature:torch.Tensor,
        up_cnn_feature:torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # todo add ac token
        # output_tokens = torch.cat([ self.ac_tokens.weight,self.cnn_tokens.weight], dim=-1)
        # output_tokens = torch.cat([ self.ac_tokens.weight], dim=-1)


        # one_stage_masks = self.mask_downscaling(one_stage_masks)
        # one_stage_masks = one_stage_masks.expand(-1,self.transformer_dim,-1,-1)
        # output_tokens = output_tokens.unsqueeze(0)
        ac_token =  self.ac_pred_head_sam(ac_token)

        up_feat_cnn, up_feat_l1_cnn = self.upblock(cnn_first_feature,cnn_feature,up_cnn_feature,image_embeddings)
        # up_feat_vit, up_feat_l1_vit = self.upblock(cnn_first_feature, cnn_feature, image_embeddings, mode='vit')
        # cnn_token =  self.ac_pred_head_cnn(cnn_token)

        output_tokens =  ac_token
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # TODO
        # early_embeddings = self.early_upscaling(early_embeddings)
        # cnn_features_2 = self.third_cnn_transf(cnn_features_2)
        #
        # image_embeddings = self.imgconv(image_embeddings)
        # up_cnn_feature = self.cnn_transf(up_cnn_feature)
        # image_embeddings = torch.cat((image_embeddings ,up_cnn_feature),dim=1)
        # image_embeddings = self.feat_fusion(image_embeddings)
        #
        #
        # cnn_feature = self.cnn_downdim(cnn_feature)
        # cnn_first_feature = self.first_cnn_transf(cnn_first_feature)
        # image_embeddings = torch.cat((image_embeddings , early_embeddings),dim=1)
        # image_embeddings = torch.cat((image_embeddings ,up_cnn_feature,cnn_feature),dim=1)
        # image_embeddings = self.feat_fusion(image_embeddings)
        # todo
        # aspp = self.ASPP(image_embeddings)
        # image_embeddings =  torch.cat((image_embeddings,aspp),dim=1)
        #
        # src = torch.repeat_interleave(image_embeddings, output_tokens.shape[0], dim=0)

        # b, c, h, w = src.shape

        # one_stage_masks = torch.softmax(one_stage_masks.flatten(1),dim=1)
        # one_stage_masks = one_stage_masks.view(b,-1,h,w)
        # src = ( src * one_stage_masks)

        # pos_emd = torch.zeros(1, self.transformer_dim, 256, 256, device=self.ac_tokens.weight.device)
        # pos_src = torch.repeat_interleave(one_stage_masks, output_tokens.shape[0], dim=0)

        # image_pe = torch.repeat_interleave(image_pe, 2, dim=1)
        # zero_pos = torch.zeros_like(cnn_feature,device=cnn_feature.device,requires_grad=False)
        # image_pe = torch.cat((image_pe,zero_pos),dim=1)
        # image_pe = torch.cat((zero_pos,image_pe), dim=1)
        posemb_1 = self.posemb_1.to(image_embeddings.device)
        posemb_2 = self.posemb_2.to(image_embeddings.device)
        image_pe =  [posemb_1,posemb_2]
        # cnn_feature_up = self.output_upscaling(cnn_feature)
        # image_embeddings_up = self.output_upscaling(image_embeddings)


        src_cnn = [up_feat_cnn,
               up_feat_l1_cnn]

        # src_vit = [up_feat_vit,
        #            up_feat_l1_vit]



        # todo




        # Run the transformers
        hs, src_cnn = self.transformer(src_cnn, image_pe, output_tokens)
        # _, src_vit = self.transformer(src_vit, image_pe, output_tokens)

        # todo ac token
        ac_tokens_out =  hs[:,  -1, :]
        b, c, h, w = up_feat_l1_cnn.shape
        # Upscale mask embeddings and predict masks using the mask tokens
        src_cnn = src_cnn.transpose(1, 2).view(b, c, h, w)
        # src_vit = src_vit.transpose(1, 2).view(b, c, h, w)

        # image_pe = self.output_upscaling(image_pe)




        # zero_pos = torch.zeros_like(cnn_first_feature,device=cnn_first_feature.device,requires_grad=False)
        # image_pe = torch.cat((image_pe,zero_pos),dim=1)


        # upscaled_embedding = torch.cat((upscaled_embedding,cnn_first_feature),dim=1)

        # upscaled_embedding = self.feat_fusion_2(src)



        # todo ac mask
        ac_hyper = self.ac_pred_head_1(ac_tokens_out)
        ac_hyper = ac_hyper.unsqueeze(0)
        # ac_hyper = torch.repeat_interleave(ac_hyper,4,dim=-1)

        # todo ac_masks   make sure 'multimask_output' is False
        # one_stage_mask = (ac_hyper @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # b, c, h, w = upscaled_embedding.shape

        # hs, src = self.transformer2(upscaled_embedding, image_pe, ac_hyper)
        # src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = src_cnn
        upscaled_embedding = self.feat_fusion(upscaled_embedding)
        # ac_tokens_out = hs[:, -1, :]
        # ac_hyper = self.ac_pred_head_2(ac_tokens_out)
        # ac_hyper = ac_hyper.unsqueeze(0)
        #
        #
        b, c, h, w = upscaled_embedding.shape
        # todo ac_masks   make sure 'multimask_output' is False
        ac_masks = (ac_hyper @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)





        return ac_masks


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



class Upblock(nn.Module):
    def __init__(self,
                 activation=nn.ReLU,
                 l1dim=128,
                 l2dim=256,
                 upfeatdim=32,
                 ):

        super().__init__()

        self.conv1 =  nn.Sequential(
            nn.Conv2d(upfeatdim , upfeatdim  , kernel_size=3, stride=1,padding=1, bias=True),
            LayerNorm2d(upfeatdim ),
            activation(),



        )
        self.conv_vit = nn.Sequential(
            nn.Conv2d(upfeatdim, upfeatdim, kernel_size=3, stride=1,padding=1, bias=True),
            LayerNorm2d(upfeatdim),
            activation(),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(l2dim, upfeatdim *2 , kernel_size=1, stride=1, bias=True),
            LayerNorm2d(upfeatdim *2 ),
            activation(),
            nn.Conv2d(upfeatdim*2, upfeatdim*2, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm2d(upfeatdim*2),
            activation(),

        )

        self.fusion =  nn.Sequential(
            nn.Conv2d(upfeatdim*4, upfeatdim*4,kernel_size=1, stride=1,bias=True),
            LayerNorm2d(upfeatdim*4),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(l1dim, upfeatdim * 2, kernel_size=1, stride=1, bias=True),
            LayerNorm2d(upfeatdim * 2),
            activation(),
            nn.Conv2d(upfeatdim * 2, upfeatdim * 2, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm2d(upfeatdim * 2),
            activation(),

        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(upfeatdim*4, upfeatdim*2, kernel_size=2, stride=2),
            LayerNorm2d(upfeatdim*2),
            activation(),

        )

        self.fusion_2 = nn.Sequential(
            nn.Conv2d(upfeatdim * 4, upfeatdim * 2, kernel_size=1, stride=1, bias=True),
            LayerNorm2d(upfeatdim * 2),


        )




        # # 定义 SE 注意力模块
        # self.se_attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(upfeatdim, upfeatdim // 4, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(upfeatdim // 4, upfeatdim, kernel_size=1),
        #     nn.Sigmoid()
        # )

    def forward(self, l1feat, l2feat, up_feat,vit_feat):
        """
        l1feat:level 1 scale is h//2,w//2   from backbone(vit) or cnn.
        l2feat:level 2 scale is h//4,w//4   from backbone(vit) or cnn.
        up_feat:the up_sampled feat scale is h//4,w//4
        """


        up_feat = self.conv1(up_feat)
        vit_feat = self.conv_vit(vit_feat)
        l2feat = self.conv2(l2feat)
        up_feat =  torch.cat((vit_feat,up_feat, l2feat), dim=1)
        up_feat = self.fusion(up_feat)

        up_feat_l1 = self.up1(up_feat)
        l1feat = self.conv3(l1feat)
        up_feat_l1 = torch.cat((up_feat_l1, l1feat), dim=1)

        up_feat_l1 = self.fusion_2(up_feat_l1)


        return up_feat, up_feat_l1


