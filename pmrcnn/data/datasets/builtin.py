import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_nonvoc": ("/public/home/meijilin/dataset/coco2017/train2017",
                               "coco/new_annotations/base_instances_train2017.json"),
    "coco_2017_train_voc_10_shot": ("/public/home/meijilin/dataset/coco2017/train2017",
                                    "coco/new_annotations/novel_10_shot_instances_train2017.json"),
}
_PREDEFINED_SPLITS_COCO["318"]={
    "318_base_train": ("/public/home/meijilin/dataset/318/318mixed/images/full",
                        "318_2class/new_annotations/instances_mixed_cattle_motorcyclist.json"),
    "318_novel_10_shot_train": ("/public/home/meijilin/dataset/318/318mixed/images/full",
                                "318_2class/new_annotations/instances_novel_20_shot.json"),
    "318_val": ("/home/qianhao/ZSD_tcb/data/318/cattle_all_other_view",
                "/home/qianhao/PMR-CNN/datasets/coco/new_annotations/only_cattle_all_view1.json")
    # "318_val": ("/home/qianhao/ZSD_tcb/data/318/cattle_all_other_view",
    #         "/home/qianhao/PMR-CNN/datasets/coco/new_annotations/only_cattle_all_view.json")
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
