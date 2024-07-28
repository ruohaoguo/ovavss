import logging

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.video_maskformer import VideoMaskFormer
from .modeling.clip_adapter import build_clip_adapter

from .modeling.audio_encoder import vggish
from .modeling.audio_encoder.vgg_config import vgg_cfg

logger = logging.getLogger(__name__)


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


@META_ARCH_REGISTRY.register()
class OV_AVSS(VideoMaskFormer):
    @configurable
    def __init__(self, *, clip_adapter: nn.Module, **kwargs):
        super().__init__(**kwargs)

        self.clip_adapter = clip_adapter
        self.class_names = ["accordion", "airplane", "axe", "baby", "bassoon", "bell", "bird", "boat", "boy", "bus", "car",
                            "cat", "cello", "clarinet", "clipper", "clock", "dog", "donkey", "drum", "duck", "elephant",
                            "emergency-car", "erhu", "flute", "frying-food", "girl", "goose", "guitar", "gun", "guzheng",
                            "hair-dryer", "handpan", "harmonica", "harp", "helicopter", "hen", "horse", "keyboard", "leopard",
                            "lion", "man", "marimba", "missile-rocket", "motorcycle", "mower", "parrot", "piano", "pig",
                            "pipa", "saw", "saxophone", "sheep", "sitar", "sorna", "squirrel", "tabla", "tank", "tiger",
                            "tractor", "train", "trombone", "truck", "trumpet", "tuba", "ukulele", "utv", "vacuum-cleaner",
                            "violin", "wolf", "woman"]

        self.audio_encoder = audio_extractor(vgg_cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def from_config(self, cfg):
        args_dict = VideoMaskFormer.from_config(cfg)
        # hardcode to 1
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1

        clip_adapter = build_clip_adapter(cfg.MODEL.CLIP_ADAPTER)
        # open-vocabulary
        args_dict["clip_adapter"] = clip_adapter

        return args_dict

    def forward(self, batched_inputs):
        class_names = self.class_names
        self.sem_seg_head.num_classes = len(class_names)

        images = []
        video_labels = []
        frame_id = []
        audios = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
            audios.append(video["wav_data"])
            frame_id.append([int(fn.split("/")[-1][0]) for fn in video["file_names"]])
            if self.training:
                video_labels.append(video["label"])

        # prepare audio features
        with torch.no_grad():
            audio_feats = self.audio_encoder(audios)
            a_b, a_t, a_c = len(audios), int(audio_feats.shape[0] / len(audios)), audio_feats.shape[1]
            audio_feats_extract = torch.zeros((a_b, self.num_frames, a_c))
            for i in range(a_b):
                for j in range(self.num_frames):
                    audio_feats_extract[i][j] = audio_feats[i * a_t + frame_id[i][j]]  # (1, 5, 128)
            audio_features = audio_feats_extract.reshape(a_b * self.num_frames, a_c)
            audio_feats = audio_features.cuda()

        # prepare image features
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        outputs = self.sem_seg_head(audio_feats, features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets_avsbench(batched_inputs, images)
            # remove classes
            for i in range(len(targets)):
                targets[i]['labels'] = torch.zeros_like(targets[i]['labels'])
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, video_labels)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_score = outputs["pred_logits"][0]
            mask_pred_result = outputs["pred_masks"][0]

            ph, pw = mask_pred_result.shape[-2:]
            ih, iw = images.tensor.shape[-2:]
            if ph != ih or pw != iw:
                # upsample masks
                mask_pred_result = F.interpolate(
                    mask_pred_result,
                    size=(ih, iw),
                    mode="bilinear",
                    align_corners=False,
                )

            mask_cls_result, mask_pred_result = self.open_vocabulary_inference(mask_score, mask_pred_result, torch.stack([x.to(self.device) for x in batched_inputs[0]["image"]]), class_names)

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(self.num_queries, len(class_names), mask_cls_result, mask_pred_result, image_size, height, width)

    def open_vocabulary_inference(self, scores: torch.Tensor, masks: torch.Tensor, frames: torch.Tensor, class_names):
        if len(scores) > 0:
            frame_len = len(frames)
            part_len = 5
            clip_cls = []
            valid_flag = []
            for idx in range(0, frame_len, part_len):
                part_frames = frames[idx:idx+part_len]
                part_masks = masks[:, idx:idx+part_len].sigmoid().transpose(0, 1).contiguous()
                part_clip_cls, part_valid_flag = self.clip_adapter(part_frames, class_names, part_masks)
                if part_clip_cls is None:
                    part_clip_cls = torch.empty(0, len(class_names), device=self.device)
                clip_cls.append(part_clip_cls); valid_flag.append(part_valid_flag)
            # remove non-object logits
            clip_cls = torch.cat(clip_cls)
            valid_flag = torch.cat(valid_flag)

            if torch.sum(valid_flag) == 0:
                return [], []

            # M x 2 (frame_idx, query_idx)
            valid_ids = torch.nonzero(valid_flag)
            # N
            valid_query_flag = torch.sum(valid_flag, dim=0) > 0
            # N x 1 -> N
            valid_query_ids = torch.nonzero(valid_query_flag)[:, 0]

            # frame-level average
            query_clip_cls = [torch.mean(clip_cls[valid_ids[:, 1] == query_id], dim=0) for query_id in valid_query_ids]
            clip_cls = torch.stack(query_clip_cls)

            logits = clip_cls.softmax(dim=-1)
            masks  = masks[valid_query_flag]
        else:
            logits = []
            masks = []

        return logits, masks


