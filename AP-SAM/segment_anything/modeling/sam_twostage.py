# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .mask_decoder2 import MaskDecoder2
from .common import LayerNorm2d
from .unetencoder import UNetEncoder

class Sam_Twostage(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        mask_decoder2:MaskDecoder2,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.mask_decoder2 = mask_decoder2
        # self.avgpool = nn.AvgPool2d((55,64))
        self.cnnencoder = UNetEncoder(3)


        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         1024,
        #         256,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(256),
        #     nn.Conv2d(
        #         256,
        #         32,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(32),
        # )

        # The SAM image encoder dim is 1024，and the final image feature is 256,‘32’ references RSprompter setting
        # self.feature_fusion = feature_fusion(1024,256,32)







        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        prompt_signal: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)


        # todo early_feature is first attention feature
        cnn_features,first_cnn_features,early_cnn_features = self.cnnencoder(input_images)
        image_embeddings = self.image_encoder(input_images)



        # feature_fusion = self.neck(interm_embeddings)
        # feature_fusion = self.feature_fusion(image_embeddings,interm_embeddings)


        outputs = []
        for image_record, curr_embedding,cnn_feature,early_cnn_feature,first_cnn_feature\
                in zip(batched_input, image_embeddings,cnn_features,early_cnn_features,first_cnn_features):

            # todo two-stage mask prompt,first is from threshold segmentation ,second is from one-stage SAM mask decoder

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            iou_predictions,upscaled_img_embedding,up_cnn_src , ac_token = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe() ,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                cnn_feature=None,
                # cnn_feature = fin_cnn_faeture.unsqueeze(0),
            )
            # one_stage_masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )


            # the image positional encoding is one stage mask
            origin_res_masks= self.mask_decoder2(
                image_embeddings=upscaled_img_embedding,
                early_embeddings = None,
                # early_embeddings = (curr_early_feature).unsqueeze(0),
                ac_token = ac_token,
                cnn_token = ac_token,
                # cnn_features_2=cnn_features_2.unsqueeze(0),
                cnn_feature=early_cnn_feature.unsqueeze(0),
                cnn_first_feature = first_cnn_feature.unsqueeze(0),
                up_cnn_feature = cnn_feature.unsqueeze(0),

            )
            # todo origin_masks for train

            # origin_res_masks = self.postprocess_masks(
            #     origin_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            origin_masks = origin_res_masks
            # boundary_masks = origin_res_masks[:,-1,:,:].unsqueeze(0)


            # bg_masks = origin_res_masks[:,1,:,:].unsqueeze(1)

            # origin_masks = origin_masks - bg_masks
            # origin_masks = origin_res_masks + one_stagse_masks
            # origin_masks = self.postprocess_masks(
            #     origin_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            masks = origin_masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    # "low_res_masks":low_res_masks,
                    "origin_masks": origin_masks,
                    # "boundary_masks":boundary_masks
                }
            )
            # todo masks is bool matrix
        return outputs



    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        prompt_signal: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)


        # todo early_feature is first attention feature
        cnn_features,early_cnn_features,first_cnn_features = self.cnnencoder(input_images)
        image_embeddings,interm_embeddings= self.image_encoder(input_images)



        # feature_fusion = self.neck(interm_embeddings)
        # feature_fusion = self.feature_fusion(image_embeddings,interm_embeddings)


        outputs = []
        for image_record, curr_embedding,cnn_feature,early_cnn_feature,first_cnn_feature\
                in zip(batched_input, image_embeddings,cnn_features,early_cnn_features,first_cnn_features):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            # todo two-stage mask prompt,first is from threshold segmentation ,second is from one-stage SAM mask decoder

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            iou_predictions,upscaled_img_embedding,up_cnn_src , ac_token = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe() ,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                cnn_feature=None,
                # cnn_feature = fin_cnn_faeture.unsqueeze(0),
            )
            # one_stage_masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )


            # the image positional encoding is one stage mask
            origin_res_masks= self.mask_decoder2(
                image_embeddings=upscaled_img_embedding,
                early_embeddings = None,
                # early_embeddings = (curr_early_feature).unsqueeze(0),
                ac_token = ac_token,
                cnn_token = ac_token,
                # cnn_features_2=cnn_features_2.unsqueeze(0),
                cnn_feature=early_cnn_feature.unsqueeze(0),
                cnn_first_feature = first_cnn_feature.unsqueeze(0),
                up_cnn_feature = cnn_feature.unsqueeze(0),

            )
            # todo origin_masks for train

            # origin_res_masks = self.postprocess_masks(
            #     origin_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            origin_masks = origin_res_masks
            # bg_masks = origin_res_masks[:,1,:,:].unsqueeze(1)

            # origin_masks = origin_masks - bg_masks
            # origin_masks = origin_res_masks + one_stagse_masks
            # origin_masks = self.postprocess_masks(
            #     origin_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            masks = origin_masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    # "low_res_masks":low_res_masks,
                    "origin_masks": origin_masks,
                }
            )
            # todo masks is bool matrix
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size[0], self.image_encoder.img_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size[0] - h
        padw = self.image_encoder.img_size[1] - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

class feature_fusion(nn.Module):
    def __init__(self,inchannels,hidden,outchannels):
        super().__init__()

        self.downconvs = nn.ModuleList()
        self.hidden_convs = nn.ModuleList()
        for i in range(4):
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(inchannels, hidden, 1),
                    LayerNorm2d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    LayerNorm2d(hidden),
                    nn.ReLU(inplace=True),
            )
        )

            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                    LayerNorm2d(hidden),
                    nn.ReLU(inplace=True),
                )
            )





        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, outchannels, 1),
            LayerNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, 3, padding=1),
            LayerNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, 3, padding=1),
        )


    def forward(self,interm_embeddings):
        feature_list = []
        for i,x in enumerate(interm_embeddings):
            x = self.downconvs[i](x)
            feature_list.append(x)

        x = None
        for hidden_state, hidden_conv in zip(feature_list, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual

        x = self.fusion(x)

        return x

