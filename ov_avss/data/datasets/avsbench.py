import contextlib
import io
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

__all__ = ["load_avssbench_json", "register_avssbench_instances"]


AVSSBENCH_CATEGORIES = [
    {"color": [128, 0, 0], "isthing": 1, "id": 1, "name": "accordion"},
    {"color": [0, 128, 0], "isthing": 1, "id": 2, "name": "airplane"},
    {"color": [128, 128, 0], "isthing": 1, "id": 3, "name": "axe"},
    {"color": [0, 0, 128], "isthing": 1, "id": 4, "name": "baby"},
    {"color": [128, 0, 128], "isthing": 1, "id": 5, "name": "bassoon"},
    {"color": [0, 128, 128], "isthing": 1, "id": 6, "name": "bell"},
    {"color": [128, 128, 128], "isthing": 1, "id": 7, "name": "bird"},
    {"color": [64, 0, 0], "isthing": 1, "id": 8, "name": "boat"},
    {"color": [192, 0, 0], "isthing": 1, "id": 9, "name": "boy"},
    {"color": [64, 128, 0], "isthing": 1, "id": 10, "name": "bus"},
    {"color": [192, 128, 0], "isthing": 1, "id": 11, "name": "car"},
    {"color": [64, 0, 128], "isthing": 1, "id": 12, "name": "cat"},
    {"color": [192, 0, 128], "isthing": 1, "id": 13, "name": "cello"},
    {"color": [64, 128, 128], "isthing": 1, "id": 14, "name": "clarinet"},
    {"color": [192, 128, 128], "isthing": 1, "id": 15, "name": "clipper"},
    {"color": [0, 64, 0], "isthing": 1, "id": 16, "name": "clock"},
    {"color": [128, 64, 0], "isthing": 1, "id": 17, "name": "dog"},
    {"color": [0, 192, 0], "isthing": 1, "id": 18, "name": "donkey"},
    {"color": [128, 192, 0], "isthing": 1, "id": 19, "name": "drum"},
    {"color": [0, 64, 128], "isthing": 1, "id": 20, "name": "duck"},
    {"color": [128, 64, 128], "isthing": 1, "id": 21, "name": "elephant"},
    {"color": [0, 192, 128], "isthing": 1, "id": 22, "name": "emergency-car"},
    {"color": [128, 192, 128], "isthing": 1, "id": 23, "name": "erhu"},
    {"color": [64, 64, 0], "isthing": 1, "id": 24, "name": "flute"},
    {"color": [192, 64, 0], "isthing": 1, "id": 25, "name": "frying-food"},
    {"color": [64, 192, 0], "isthing": 1, "id": 26, "name": "girl"},
    {"color": [192, 192, 0], "isthing": 1, "id": 27, "name": "goose"},
    {"color": [64, 64, 128], "isthing": 1, "id": 28, "name": "guitar"},
    {"color": [192, 64, 128], "isthing": 1, "id": 29, "name": "gun"},
    {"color": [64, 192, 128], "isthing": 1, "id": 30, "name": "guzheng"},
    {"color": [192, 192, 128], "isthing": 1, "id": 31, "name": "hair-dryer"},
    {"color": [0, 0, 64], "isthing": 1, "id": 32, "name": "handpan"},
    {"color": [128, 0, 64], "isthing": 1, "id": 33, "name": "harmonica"},
    {"color": [0, 128, 64], "isthing": 1, "id": 34, "name": "harp"},
    {"color": [128, 128, 64], "isthing": 1, "id": 35, "name": "helicopter"},
    {"color": [0, 0, 192], "isthing": 1, "id": 36, "name": "hen"},
    {"color": [128, 0, 192], "isthing": 1, "id": 37, "name": "horse"},
    {"color": [0, 128, 192], "isthing": 1, "id": 38, "name": "keyboard"},
    {"color": [128, 128, 192], "isthing": 1, "id": 39, "name": "leopard"},
    {"color": [64, 0, 64], "isthing": 1, "id": 40, "name": "lion"},
    {"color": [192, 0, 64], "isthing": 1, "id": 41, "name": "man"},
    {"color": [64, 128, 64], "isthing": 1, "id": 42, "name": "marimba"},
    {"color": [192, 128, 64], "isthing": 1, "id": 43, "name": "missile-rocket"},
    {"color": [64, 0, 192], "isthing": 1, "id": 44, "name": "motorcycle"},
    {"color": [192, 0, 192], "isthing": 1, "id": 45, "name": "mower"},
    {"color": [64, 128, 192], "isthing": 1, "id": 46, "name": "parrot"},
    {"color": [192, 128, 192], "isthing": 1, "id": 47, "name": "piano"},
    {"color": [0, 64, 64], "isthing": 1, "id": 48, "name": "pig"},
    {"color": [128, 64, 64], "isthing": 1, "id": 49, "name": "pipa"},
    {"color": [0, 192, 64], "isthing": 1, "id": 50, "name": "saw"},
    {"color": [128, 192, 64], "isthing": 1, "id": 51, "name": "saxophone"},
    {"color": [0, 64, 192], "isthing": 1, "id": 52, "name": "sheep"},
    {"color": [128, 64, 192], "isthing": 1, "id": 53, "name": "sitar"},
    {"color": [0, 192, 192], "isthing": 1, "id": 54, "name": "sorna"},
    {"color": [128, 192, 192], "isthing": 1, "id": 55, "name": "squirrel"},
    {"color": [64, 64, 64], "isthing": 1, "id": 56, "name": "tabla"},
    {"color": [192, 64, 64], "isthing": 1, "id": 57, "name": "tank"},
    {"color": [64, 192, 64], "isthing": 1, "id": 58, "name": "tiger"},
    {"color": [192, 192, 64], "isthing": 1, "id": 59, "name": "tractor"},
    {"color": [64, 64, 192], "isthing": 1, "id": 60, "name": "train"},
    {"color": [192, 64, 192], "isthing": 1, "id": 61, "name": "trombone"},
    {"color": [64, 192, 192], "isthing": 1, "id": 62, "name": "truck"},
    {"color": [192, 192, 192], "isthing": 1, "id": 63, "name": "trumpet"},
    {"color": [32, 0, 0], "isthing": 1, "id": 64, "name": "tuba"},
    {"color": [160, 0, 0], "isthing": 1, "id": 65, "name": "ukulele"},
    {"color": [32, 128, 0], "isthing": 1, "id": 66, "name": "utv"},
    {"color": [160, 128, 0], "isthing": 1, "id": 67, "name": "vacuum-cleaner"},
    {"color": [32, 0, 128], "isthing": 1, "id": 68, "name": "violin"},
    {"color": [160, 0, 128], "isthing": 1, "id": 69, "name": "wolf"},
    {"color": [32, 128, 128], "isthing": 1, "id": 70, "name": "woman"},
]



def _get_avsbench_instances_meta():
    thing_ids = [k["id"] for k in AVSSBENCH_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in AVSSBENCH_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 70, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in AVSSBENCH_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def load_avssbench_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from .avsbench_api.avsbench_anno import AVSBench

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        avis_api = AVSBench(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(avis_api.getCatIds())
        cats = avis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(avis_api.vids.keys())
    vids = avis_api.loadVids(vid_ids)

    anns = [avis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(avis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in AVSBench format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        record["label"] = vid_dict["label"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if record["label"] == "v1s":
                    if not (_bboxes and _segm and _bboxes[0] and _segm[0]):
                        continue

                    if frame_idx == 0:
                        bbox = _bboxes[0]
                        segm = _segm[0]
                    else:
                        bbox = []
                        segm = []
                else:
                    if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                        continue

                    bbox = _bboxes[frame_idx]
                    segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs

        # get audio pth
        audios_path = "./datasets/AVSBench-semantic/"
        audio_pth = os.path.join(audios_path, record["label"], vid_dict['file_names'][0].split("/")[0], 'audio.wav')
        record["wav_data"] = audio_pth

        dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. ".format(
                    num_instances_without_valid_segmentation
                )
                + "There might be issues in your dataset generation process. "
                "A valid polygon should be a list[float] with even length >= 6."
            )
    return dataset_dicts


def register_avssbench_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_avssbench_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="avis", **metadata
    )
