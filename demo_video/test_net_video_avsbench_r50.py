import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import argparse
import multiprocessing as mp

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import numpy as np

from torch.cuda.amp import autocast
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from ov_avss import add_maskformer2_video_config, add_maskformer2_config, add_open_vocabulary_config
from predictor import VisualizationDemo

from avsbench_dataloader import AVSBenchTest
from avsbench_utils import save_and_compute
from avsbench_eval import scores_gzsl

import warnings
warnings.filterwarnings("ignore")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_open_vocabulary_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="model test")
    parser.add_argument("--config-file", default="./configs/avsbench/OV_AVSS_R50.yaml")
    parser.add_argument("--model_input", default="./pre_models/")
    parser.add_argument("--audio_input", default="./datasets/AVSBench-semantic/")
    parser.add_argument("--output", default="./output_ov_avss_r50/result/")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    num_frames = args.num_frames

    pths = ["model_ov_avss_r50.pth"]

    for f in pths:
        print("=========> {}".format(f))

        cfg["MODEL"]["WEIGHTS"] = args.model_input + f
        demo = VisualizationDemo(cfg)

        output_path = os.path.join(args.output, f.split(".")[0])
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        # overall cmIoU & F score
        miou_pc = torch.zeros((71))  # miou value per class (total sum)
        Fs_pc = torch.zeros((71))  # f-score per class (total sum)
        cls_pc = torch.zeros((71))  # count per class

        gt_masks_all = []
        pred_masks_all = []

        test_dataset = AVSBenchTest()
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        for video_n, batch_data in enumerate(test_dataloader):
            imgs_pth, annos, video_name, imgs_names, audios = batch_data
            assert len(imgs_pth) == 5 or len(imgs_pth) == 10

            gt_masks_all = gt_masks_all + [np.array(sublist) for sublist in annos]

            for image_n in range(0, len(imgs_pth), num_frames):
                frames_pth = imgs_pth[image_n:image_n + num_frames]
                vid_annos = annos[image_n:image_n + num_frames]
                vid_frames_name = imgs_names[image_n:image_n + num_frames]
                vid_frames_name_id = [ss[0].split(".")[0] for ss in vid_frames_name]

                vid_frames = []
                for img_pth in frames_pth:
                    img = read_image(img_pth[0], format="BGR")
                    vid_frames.append(img)

                with autocast():
                    predictions_all_labels, predictions_all_masks = demo.run_on_video_avsbench(vid_frames, vid_frames_name_id, audios[0], args.confidence_threshold)
                    save_pth = os.path.join(output_path, video_name[0])
                    h, w = vid_frames[0].shape[0], vid_frames[0].shape[1]
                    _miou_pc, _fscore_pc, _cls_pc, pred_masks = save_and_compute(predictions_all_labels, predictions_all_masks, save_pth, vid_frames_name, vid_annos)

                    pred_masks = pred_masks.int()
                    pred_masks_all = pred_masks_all + [np.array(pred_masks[ii]) for ii in range(pred_masks.shape[0])]

                    # compute miou, J-measure
                    miou_pc += _miou_pc
                    cls_pc += _cls_pc
                    # compute f-score, F-measure
                    Fs_pc += _fscore_pc

            batch_iou = miou_pc / cls_pc
            batch_iou[torch.isnan(batch_iou)] = 0
            batch_iou = torch.sum(batch_iou) / torch.sum(cls_pc != 0)
            batch_fscore = Fs_pc / cls_pc
            batch_fscore[torch.isnan(batch_fscore)] = 0
            batch_fscore = torch.sum(batch_fscore) / torch.sum(cls_pc != 0)

        miou_pc = miou_pc / cls_pc
        miou_pc[torch.isnan(miou_pc)] = 0
        miou = torch.mean(miou_pc).item()
        miou_noBg = torch.mean(miou_pc[:-1]).item()
        f_score_pc = Fs_pc / cls_pc
        f_score_pc[torch.isnan(f_score_pc)] = 0
        f_score = torch.mean(f_score_pc).item()
        f_score_noBg = torch.mean(f_score_pc[:-1]).item()

        results, _ = scores_gzsl(gt_masks_all, pred_masks_all, 71,
                    [2,4,6,7,8,9,10,11,12,16,17,18,19,20,21,26,27,28,29,31,35,36,37,38,40,41,43,44,46,47,48,52,55,58,59,60,62,63,67,70],
                    [1,3,5,13,14,15,22,23,24,25,30,32,33,34,39,42,45,49,50,51,53,54,56,57,61,64,65,66,68,69])

        print("m_iou: {:.4f} || m_iou_seen: {:.4f} || m_iou_unseen: {:.4f} || m_iou_harmonic: {:.4f}".format(
            results["Mean IoU"], results["Mean IoU Seen"], results["Mean IoU Unseen"], results["Mean IoU Harmonic"]))
