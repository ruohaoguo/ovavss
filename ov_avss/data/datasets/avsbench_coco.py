import os

from .avsbench import (
    register_avssbench_instances,
    _get_avsbench_instances_meta,
)

# ==== Predefined splits for AVSBench ===========
_PREDEFINED_SPLITS_AVSBENCH = {
    "avsbench_ov_train_base": ("AVSBench-openvoc/train/JPEGImages",
                               "AVSBench-openvoc/avsbench_ov_train_base.json"),
    "avsbench_ov_val_all": ("AVSBench-openvoc/val/JPEGImages",
                            "AVSBench-openvoc/avsbench_ov_val_base_novel.json"),
}

def register_all_avsbench(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_AVSBENCH.items():
        register_avssbench_instances(
            key,
            _get_avsbench_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_avsbench(_root)
