# Modified by Zesen Cheng from https://github.com/sukjunhwang/IFC

import os
import copy
import itertools
import json
import logging
from collections import OrderedDict

import torch
import numpy as np
import pycocotools.mask as mask_util

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from .bursteval import BURSTesval


class BURSTEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        
        gt_json_path = PathManager.get_local_path(self._metadata.json_file)

        assert self._output_dir is not None, "Burst evaluation only supports offline mode."
        self._infer_json_path = os.path.join(self._output_dir, 'results.json') 
        self._burst_api = BURSTesval(gt_json_path, self._infer_json_path)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_burst_json_video(inputs, outputs)
        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for BURST format ...")

        # unmap the category ids for BURST
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for idx, pred in enumerate(predictions):
                track_category_ids = pred["track_category_ids"]
                for track_id, category_id in track_category_ids.items():
                    assert category_id < num_classes, (
                        f"A prediction has class={category_id}, "
                        f"but the dataset only has {num_classes} classes and "
                        f"predicted class id should be in [0, {num_classes - 1}]."
                    )
                    pred['track_category_ids'][track_id] = reverse_id_mapping[category_id]

        burst_predictions = {'sequences': predictions}
        if self._output_dir:
            file_path = self._infer_json_path
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(burst_predictions))
                f.flush()

        res = self._burst_api.evaluate()
        for k, v in res.items():
            self._results[k + "_segm"] = v


def instances_to_burst_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a BURST-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    burst_results = {}
    for k in ['width', 'height', 'seq_name', 'dataset', 'annotated_image_paths']:
        burst_results[k] = inputs[0][k]
    video_length = inputs[0]["length"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    entropys = outputs["pred_entropys"]

    burst_results['segmentations'] = [{} for _ in range(video_length)]
    burst_results['track_category_ids'] = {}

    for instance_id, (s, l, m, e) in enumerate(zip(scores, labels, masks, entropys)):
        valid_ids = torch.nonzero(m.sum((-2, -1)) > 20)
        for valid_id in valid_ids[:, 0]:
            rle = mask_util.encode(np.array(m[valid_id][:, :, None], order="F", dtype="uint8"))[0]
            rle = rle["counts"].decode("utf-8")
            burst_results['segmentations'][valid_id].update({
                instance_id: {
                    'rle': rle,
                    'is_gt': False,
                    'score': s,
                    'entropy': e,
                }
            })
            burst_results['track_category_ids'].update({
                instance_id: l
            })

    """
        NOTE: requirement attribute example: {
            'sequences':[
            {
                'width': 640,
                'height': 480,
                'seq_name': 'v_25685519b728afd746dfd1b2fe77c', 
                'dataset': 'YFCC100M',
                # len(annotated_image_paths) == len(segmentations)
                'annotated_image_paths': ['frame0781.jpg', 'frame0811.jpg', 'frame0841.jpg'],
                'segmentations': [[{track1 anno}, {track2 anno}], [{track1 anno}, {track2 anno}], [{track1 anno}]],
                'track_category_ids': {'1': 805, '2': 805},
            },
        {
                ...  
        },
            ...
        }
    """

    return [burst_results]
