import os
import json
import torch
import numpy as np
from PIL import Image

from avsbench_eval import calc_color_miou_fscore, scores_gzsl


def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls = 71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    label = np.argmax(semantic_map, axis=-1)
    return label


def load_color_mask_in_PIL_to_Tensor(path, v_pallete, mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    color_label = color_label.unsqueeze(0)
    return color_label   # both [1, H, W]


def save_and_compute(predictions_all_labels, predictions_all_masks, save_base_path, vid_frames_name, gt_masks):
    v_pallete = get_v2_pallete("./datasets/AVSBench-semantic/label2idx.json")
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    h = predictions_all_masks[0][0].shape[0]
    w = predictions_all_masks[0][0].shape[1]
    pred_masks = torch.zeros((len(predictions_all_masks), h, w))

    predictions_all_masks_list = []
    for iii in predictions_all_masks:
        predictions_all_masks_list_1 = []
        for jjj in iii:
            predictions_all_masks_list_1.append(jjj)
        predictions_all_masks_list.append(predictions_all_masks_list_1)

    for i in range(len(predictions_all_masks_list)):
        if predictions_all_masks_list[i] == []:
            pred_masks[i] = torch.zeros((h, w))
        else:
            for j in range(len(predictions_all_masks_list[i])):
                predictions_all_masks_list[i][j] = predictions_all_masks_list[i][j].int() * (predictions_all_labels[j] + 1)
            if len(predictions_all_masks_list[0]) < 2:
                pred_masks[i] = predictions_all_masks_list[i][0]
            else:
                result = predictions_all_masks_list[i][0]
                for mask in predictions_all_masks_list[i][1:]:
                    mask_merge = mask * (result == 0)
                    result += mask_merge
                pred_masks[i] = result

    pred_rgb_masks = np.zeros((pred_masks.shape + (3,)), np.uint8)  # [T, H, W, 3]
    for cls_idx in range(71):
        rgb = v_pallete[cls_idx]
        pred_rgb_masks[pred_masks == cls_idx] = rgb

    for idx in range(len(vid_frames_name)):
        frame_name = vid_frames_name[idx]
        frame_mask = pred_rgb_masks[idx]  # [5, 224, 224, 3]

        output_name = "%s.png" % (frame_name[0].split(".")[0])
        im = Image.fromarray(frame_mask)  # .convert('RGB')
        im.save(os.path.join(save_base_path, output_name), format='PNG')

    up = torch.nn.Upsample(size=(224, 224), mode="nearest")
    pred_masks_up = up(pred_masks.unsqueeze(dim=1))
    pred_masks_up = pred_masks_up.squeeze()
    gt_masks = torch.stack(gt_masks, dim=0).squeeze()
    _miou_pc, _fscore_pc, _cls_pc = calc_color_miou_fscore(pred_masks_up, gt_masks)

    return _miou_pc, _fscore_pc, _cls_pc, pred_masks_up
