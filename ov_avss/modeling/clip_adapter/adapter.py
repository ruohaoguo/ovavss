# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Zesen Cheng from
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/modeling/clip_adapter/adapter.py

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.transforms.transforms import Normalize
from detectron2.structures import BitMasks

import clip
from clip.model import CLIP

from .utils import build_clip_model

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


def freeze_params(model, frozen_exclude=[]):
    if "all" in frozen_exclude:
        return
    for name, param in model.named_parameters():
        if not any([exclude in name for exclude in frozen_exclude]):
            param.requires_grad = False

    return model


class ClipAdapter(nn.Module):

    def __init__(self, clip_model_name: str, text_templates: List[str] = ["a photo of {}"]):
        super().__init__()
        clip_model: CLIP = build_clip_model(clip_model_name)

        self.input_resolution = clip_model.visual.input_resolution
        self.clip_model = clip_model

        freeze_params(self.clip_model)

        self.templates = text_templates
        self.text_cache = {}

        self.mask_fill = [255.0 * c for c in PIXEL_MEAN]

        # normalize
        self.clip_prep_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        self.register_buffer("pixel_mean", torch.Tensor(PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0)
        self.register_buffer("pixel_std", torch.Tensor(PIXEL_STD).reshape(1, 3, 1, 1) * 255.0)

    def forward(self, frames: torch.Tensor, text: List[str], masks: torch.Tensor):
        """
        Args:
            frames: shape (T, C, H, W)
            masks:  shape (T, N, H, W)
            text: list[str] ["person", "dog", ...] 
        """
        frames, valid_flag = self._preprocess_image(frames, masks)
        if frames is None:
            return None, valid_flag

        frame_features = self.encode_image(frames)
        text_feature = self.encode_text(text)  # k, feat_dim
        sim_logits = self.cal_sim_logits(text_feature, frame_features)

        return sim_logits, valid_flag

    def _preprocess_image(self, frames: torch.Tensor, masks: torch.Tensor):
        """crop, mask and normalize the image.

        Args:
            frames: shape (T, C, H, W)
            masks:  shape (T, N, H, W)
        Outputs:
            regions: shape ()
        """
        masks = masks.to(frames.device)

        # 1. select valid regions
        bin_masks = masks > 0.5
        valid = bin_masks.sum(dim=(-1, -2)) > 0
        if torch.sum(valid) == 0:
            return None, valid
        valid_bin_masks = bin_masks[valid]
        valid_masks = masks[valid]

        # 2. convert masks to crop boxes
        valid_bin_masks = BitMasks(valid_bin_masks)
        valid_bboxes = valid_bin_masks.get_bounding_boxes()
        valid_bboxes = valid_bboxes.tensor.cuda().contiguous()
        # xyxy -> xywh (square crop bbox)
        sboxes = valid_bboxes.clone()
        sboxes[:, 2] = sboxes[:, 2] - sboxes[:, 0]
        sboxes[:, 3] = sboxes[:, 3] - sboxes[:, 1]
        sboxes[:, 3] = sboxes[:, 2] = torch.max(sboxes[:, 2], sboxes[:, 3])   # square ?
        # sboxes[:, 3], sboxes[:, 2] =  valid_masks.shape[1], valid_masks.shape[2]   # original ?
        sboxes[:, 2] = sboxes[:, 0] + sboxes[:, 2]
        sboxes[:, 3] = sboxes[:, 1] + sboxes[:, 3]

        # 3. crop frame region and mask region
        ids = torch.nonzero(valid).to(frames.device)
        # [frame_id, x, y, l, l]
        ind_boxes = torch.cat([ids[:, 0:1], sboxes], dim=-1)
        regions = roi_align(frames.half(), ind_boxes.half(), output_size=(self.input_resolution, self.input_resolution))
        # [query_id, x, y, l, l]
        ind_boxes = torch.cat([torch.arange(len(valid_bboxes), device=frames.device)[:, None], sboxes], dim=-1)
        mask_regions = roi_align(valid_masks.half()[:, None], ind_boxes.half(), output_size=(self.input_resolution, self.input_resolution))

        # 4. blend frame region and mask region
        regions = mask_regions * regions + (1 - mask_regions) * 0.

        return regions, valid

    def normalize(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)

    def encode_text(self, noun_list: List[str]):
        new_words = [word for word in noun_list if word not in self.text_cache]
        if len(new_words) > 0:
            text_embeds_bucket = []
            for template in self.templates:
                noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
                text_inputs = torch.cat(noun_tokens).to(self.clip_model.text_projection.data.device)
                text_embeds = self.clip_model.encode_text(text_inputs)
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                text_embeds_bucket.append(text_embeds)
            # ensemble by averaging
            text_embeds = torch.stack(text_embeds_bucket).mean(dim=0)
            text_embeds = self.normalize(text_embeds)
            self.text_cache.update(dict(zip(new_words, text_embeds)))

        cat_embeds = torch.stack([self.text_cache[word] for word in noun_list])

        return cat_embeds

    def encode_image(self, image: torch.Tensor):
        image = F.interpolate(image / 255., (self.input_resolution, self.input_resolution), mode="bicubic")
        image = self.clip_prep_img(image)
        image_features = self.clip_model.visual(image)
        return self.normalize(image_features)

    def cal_sim_logits(self, text_features: torch.Tensor, image_features: torch.Tensor, temperature: float = 100):
        return temperature * image_features @ text_features.T


class BgClipAdapter(ClipAdapter):

    def __init__(self, clip_model_name: str, text_templates: List[str]):
        super().__init__(clip_model_name, text_templates)
        self.non_object_embedding = nn.Parameter(torch.empty(1, self.clip_model.text_projection.shape[-1]))
        nn.init.normal_(self.non_object_embedding.data, std=self.clip_model.transformer.width ** -0.5)

    def encode_text(self, noun_list: List[str]):
        object_text_embeds = super().encode_text(noun_list)
        non_object_text_embeds = (self.non_object_embedding / self.non_object_embedding.norm(dim=-1, keepdim=True))
        overall_text_embeds = torch.cat([object_text_embeds, non_object_text_embeds], dim=0)
        return overall_text_embeds
