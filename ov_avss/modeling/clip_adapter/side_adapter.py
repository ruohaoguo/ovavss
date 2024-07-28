from typing import  List

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import CLIP, ResidualAttentionBlock, VisionTransformer
from torchvision.transforms.transforms import Normalize

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


def downsample2d(src, target_shape, method="nearest"):
    # src: [N,C,H,W]
    # target_shape: [H',W']
    # return: [N,C,H',W']
    if method in ["bicubic", "bilinear", "nearest"]:
        src = F.interpolate(src, size=target_shape, mode=method, align_corners=False)
    elif method == "avg":
        src = F.adaptive_avg_pool2d(src, output_size=target_shape)
    elif method == "max":
        src = F.adaptive_max_pool2d(src, output_size=target_shape)
    return src


def resize_pos_embed2d(
    posemb,
    src_shape,
    tgt_shape,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=False,
):
    """interpolate positional embedding from src_shape to tgt_shape. posemb: [N,L,C]"""
    if src_shape == tgt_shape:
        return posemb
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:])
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    posemb_grid = posemb_grid.permute(0, 2, 1).reshape(1, -1, src_shape[0], src_shape[1])

    posemb_grid = F.interpolate(
        posemb_grid,
        size=tgt_shape,
        mode=interpolation,
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, tgt_shape[0] * tgt_shape[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class BiasedResidualAttentionBlock(ResidualAttentionBlock):

    def attention(self, x: torch.Tensor, attn_mask):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class SideAdapter(nn.Module):

    def __init__(self, clip_model_name="ViT-B/16", out_dims=256, broken_idx=9, merge_ids=[3, 6, 9], num_queries=100, text_templates: List[str] = ["a photo of {}"]):
        super().__init__()
        clip_model: CLIP = build_clip_model(clip_model_name)

        self.input_resolution = clip_model.visual.input_resolution
        self.vis_width  = clip_model.visual.proj.shape[0]
        self.embed_dims = clip_model.visual.proj.shape[1]
        self.num_heads  = self.vis_width // 64
        self.grid_size  = round((clip_model.visual.positional_embedding.shape[0] - 1) ** 0.5)
        self.num_layers = len(clip_model.visual.transformer.resblocks)

        self.clip_model = clip_model

        visual_state_dict = clip_model.visual.state_dict()
        resblocks = nn.Sequential(*[BiasedResidualAttentionBlock(self.vis_width, self.num_heads) for _ in range(self.num_layers)])
        self.clip_model.visual.transformer.resblocks = resblocks
        self.clip_model.visual.load_state_dict(visual_state_dict)

        freeze_params(self.clip_model)
        self.broken_idx = broken_idx
        self.merge_ids  = merge_ids

        self.out_dims   = out_dims
        self.sos_token_num = num_queries

        self.attn_projs = nn.ModuleList()
        if self.out_dims is not None:
            for _ in range(len(self.merge_ids)):
                self.attn_projs.append(nn.Conv2d(self.vis_width, self.out_dims, kernel_size=1))
                nn.init.kaiming_uniform_(self.attn_projs[-1].weight, a=1)
                nn.init.constant_(self.attn_projs[-1].bias, 0)
        else:
            [self.attn_projs.append(nn.Identity()) for _ in range(len(self.merge_ids))]

        self.templates = text_templates
        self.bg_embed = nn.Parameter(torch.randn(1, self.embed_dims))
        nn.init.normal_(self.bg_embed, std=self.bg_embed.shape[1] ** -0.5)

        self.text_cache = {}

        # normalize
        self.clip_prep_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def forward(self, frames, text, masks):
        """
        Args:
            frames: shape (T, C, H, W)
            masks:  shape (T, Q, H, W)
            text: list[str] ["person", "dog", ...] 
        Outputs:
            logits: shape (T, N, C)
        """

        # [T, Q, H, W] -> [T, N, Q, H, W]
        attn_biases = masks[:, None].repeat(1, self.num_heads, 1, 1, 1)
        new_attn_biases = torch.zeros_like(attn_biases)
        new_attn_biases = new_attn_biases.masked_fill(attn_biases < 0, float("-inf"))
        mg_feats, bk_feats = self.front_encode_image(frames)
        frame_feats = self.post_encode_image(bk_feats, new_attn_biases)
        text_feats = self.encode_text(text, w_bg=False)  # k, feat_dim
        sim_logits = self.cal_sim_logits(text_feats, frame_feats)

        return sim_logits

    def front_encode_image(self, x: torch.Tensor):
        vis: VisionTransformer = self.clip_model.visual

        x = F.interpolate(x / 255., (self.input_resolution, self.input_resolution), mode="bicubic")
        x = self.clip_prep_img(x)

        x = vis.conv1(x)  # shape = [*, width, grid, grid]
        b, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()       # shape = [*, grid ** 2, width]
        x = torch.cat([vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos_embed = resize_pos_embed2d(vis.positional_embedding.to(x.dtype)[None, ...], (self.grid_size, self.grid_size), (h, w))[0]
        x = x + pos_embed
        x = vis.ln_pre(x)

        x = x.permute(1, 0, 2).contiguous()  # NLD -> LND
        resblocks = vis.transformer.resblocks[:self.broken_idx]

        outputs = [(x[0:1], x[1:].permute(1, 2, 0).reshape(b, -1, h, w).contiguous())]
        for i, resblock in enumerate(resblocks, start=1):
            x = resblock(x)
            outputs.append((x[0:1], x[1:].permute(1, 2, 0).reshape(b, -1, h, w).contiguous()))

        mg_feats = [f[1] for i, f in enumerate(outputs) if i in self.merge_ids]
        mg_feats = [self.attn_projs[i](f) for i, f in enumerate(mg_feats)]
        feats = outputs[-1]

        return mg_feats, feats

    def post_encode_image(self, feats, attn_bias):
        cm = self.clip_model
        # construct clip shadow features.
        cls_token = feats[0]  # 1,n,c
        pix_feat  = feats[1]  # n,c,h,w

        n, c, h, w = pix_feat.shape
        x = torch.cat([cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)])  # 1+l,n,c

        # construct sos token.
        sos_token = cls_token.repeat(self.sos_token_num, 1, 1)

        resblocks = cm.visual.transformer.resblocks[self.broken_idx:]

        if attn_bias is not None:
            if not isinstance(attn_bias, (list, tuple)):
                attn_bias = [attn_bias]
            # construct attn biases.
            attn_biases = self._build_attn_biases(attn_bias, self.num_heads, len(resblocks), target_shape=(h, w))
        else:
            attn_biases = [None] * len(resblocks)

        x = torch.cat([sos_token, x], dim=0)
        for i, resblock in enumerate(resblocks):
            x = resblock(x, attn_mask=attn_biases[i])
        sos_token = x[: self.sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD

        sos_token = cm.visual.ln_post(sos_token)
        sos_token = sos_token @ cm.visual.proj
        sos_token = F.normalize(sos_token, dim=-1)

        return sos_token

    def encode_text(self, x: List[str], w_bg=True):
        cm = self.clip_model

        x = [word.replace('(', '').replace(')', '').replace('_', ' ') for word in x]
        new_words = [word for word in x if word not in self.text_cache]
        if len(new_words) > 0:
            text_embeds_bucket = []
            for template in self.templates:
                noun_tokens = clip.tokenize([template.format(noun) for noun in new_words])
                text_inputs = noun_tokens.to(self.bg_embed.device)
                text_embeds = cm.encode_text(text_inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                text_embeds_bucket.append(text_embeds)
            text_embeds = torch.stack(text_embeds_bucket).mean(dim=0)
            text_embeds = F.normalize(text_embeds, dim=-1)
            self.text_cache.update(dict(zip(new_words, text_embeds)))

        cat_embeds = torch.stack([self.text_cache[word] for word in x])
        if w_bg:
            cat_embeds = torch.cat([cat_embeds, F.normalize(self.bg_embed, dim=-1)], dim=0)

        return cat_embeds

    def cal_sim_logits(self, text_feats: torch.Tensor, image_feats: torch.Tensor):
        return self.clip_model.logit_scale.exp() * image_feats @ text_feats.T

    def _build_attn_biases(self, attn_biases, num_heads, num_layers, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: B*num_head,L,L
            # attn_bias: [B, num_head, num_sos, H, W]
            b, num_head, num_sos, h, w = attn_bias.shape
            # 1. reshape and downsample
            attn_bias = downsample2d(attn_bias.reshape(b, num_head * num_sos, h, w), target_shape, method='max')
            attn_bias = attn_bias.reshape(b, num_head, num_sos, *target_shape)
            true_num_head = num_heads
            assert (num_head == 1 or num_head == true_num_head), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(b * true_num_head, num_sos, -1)
            # L = tH x tW
            L = attn_bias.shape[-1]
            # 2. prepare attn biases
            # [n*num_head, num_sos+1+L, num_sos+1+L]
            new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
            # cut interaction between patch tokens and mask tokens.
            new_attn_bias[:, :num_sos] = -100
            # cut interaction between cls token and mask tokens. 
            new_attn_bias[:num_sos, num_sos] = -100
            # provide interaction between mask token and itself. 
            new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
            # [num_sos+1+L, num_sos+1+L] -> [B*num_head, num_sos+1+L, num_sos+1+L]
            new_attn_bias = (new_attn_bias[None, ...].expand(b * true_num_head, -1, -1).clone())
            # 3. insert condition into attn biases between sos tokens and patch tokens
            new_attn_bias[..., :num_sos, -L:] = attn_bias
            formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in range(num_layers)]
        return formatted_attn_biases
