#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.data import build_batch_data_loader

from pmrcnn.config import get_cfg
from pmrcnn.data.dataset_mapper import DatasetMapperWithSupport
from pmrcnn.data.build import build_detection_train_loader, build_detection_test_loader
from pmrcnn.solver import build_optimizer
from pmrcnn.evaluation import COCOEvaluator

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
from typing import List, Union
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators
from detectron2.utils.comm import get_world_size
import time, datetime
from collections import abc
from contextlib import ExitStack
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import Instances, Boxes
from detectron2.modeling.postprocessing import detector_postprocess
import onnxruntime

class OnnxModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        support_file_name = '/home/qianhao/PMR-CNN/support_dir/support_feature.pkl'
        self.init_support(support_file_name=support_file_name)

    def init_support(self, support_file_name):
        with open(support_file_name, "rb") as hFile:
            self.support_dict = pickle.load(hFile, encoding="latin1")
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

    def forward(self, image):
        image = ((image - self.model.pixel_mean) / self.model.pixel_std).unsqueeze(0)
        features = self.model.backbone(image)
        # B, _, _, _ = features['res4'].shape
        support_proposals_dict = {}
        support_box_features_dict = {}
        proposal_num_dict = {}
        query_images = ImageList.from_tensors([image])  # one query image

        query_features_res4 = features['res4']  # one query feature for attention rpn
        query_features = {'res4': query_features_res4}  # one query feature for rcnn
        for cls_id, res4_avg in self.support_dict['res4_avg'].items():
            # support branch ##################################
            support_box_features = self.support_dict['res5_avg'][cls_id]
            support_box_features_in = self.model.layer5(support_box_features)

            # PMMs
            prototype_list, Prob_map = self.model.PMMs(support_box_features_in, query_features)

            feature_size = query_features['res4'].shape[-2:]
            for j in range(self.model.num_pro):

                vec = prototype_list[j]
                exit_feat_in_ = self.model.f_v_concate(query_features['res4'], vec, feature_size)
                exit_feat_in_ = self.model.layer5(exit_feat_in_)
                if j == 0:
                    exit_feat_in = exit_feat_in_
                else:
                    exit_feat_in = exit_feat_in + exit_feat_in_
            exit_feat = self.model.layer6(exit_feat_in)

            # concat
            pos_concat = torch.cat([exit_feat, Prob_map], dim=1)
            # pos_conv = self.layer7(pos_concat)

            pos_features = {'res4': exit_feat}
            support_correlation = pos_features  # attention map for attention rpn

            proposals, _ = self.model.proposal_generator(query_images, support_correlation, None)
            support_proposals_dict[cls_id] = proposals
            support_box_features_dict[cls_id] = support_box_features

            if cls_id not in proposal_num_dict.keys():
                proposal_num_dict[cls_id] = []
            proposal_num_dict[cls_id].append(len(proposals[0]))
        bbox, score, pred_cls, image_shape = self.model.roi_heads.eval_with_support_4onnx(query_images, query_features, support_proposals_dict,
                                                      support_box_features_dict)
        box_num = bbox.shape[0]
        # bbox_pad = torch.zeros([100,4], dtype=bbox.dtype, device=bbox.device)
        # score_pad = torch.zeros([100], dtype=score.dtype, device=score.device)
        # pred_cls_pad = torch.zeros([100], dtype=pred_cls.dtype, device=pred_cls.device)
        # bbox_pad[:bbox.shape[0], :bbox.shape[1]] = bbox
        # score_pad[:score.shape[0]] = score
        # pred_cls_pad[:pred_cls.shape[0]] = pred_cls
        return bbox, score, pred_cls, image_shape, torch.tensor(box_num, dtype=torch.int32)

class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithSupport(cfg)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    dummy_inputs = None
    with torch.no_grad():
        if isinstance(model, nn.Module):
            model.eval()
            model.to('cuda')
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            img = inputs[0]['image'].to('cuda')
            width = inputs[0]['width']
            height = inputs[0]['height']
            if dummy_inputs == None:
                dummy_inputs = img
            if isinstance(model, nn.Module):
                bbox, score, pred_cls, image_shape, box_num = model(img)
                box_num = box_num.item()
                bbox = bbox[:box_num]
                score = score[:box_num]
                pred_cls = pred_cls[:box_num]
            elif isinstance(model, onnxruntime.InferenceSession):
                ort_input = {"image": img.cpu().numpy()}
                bbox, score, pred_cls, image_shape, box_num = model.run(None, ort_input)
                box_num = box_num.item()
                bbox = bbox[:box_num]
                score = score[:box_num]
                pred_cls = pred_cls[:box_num]
                bbox = torch.from_numpy(bbox)
                score = torch.from_numpy(score)
                pred_cls = torch.from_numpy(pred_cls)
                image_shape = torch.from_numpy(image_shape)
            result = Instances(tuple(image_shape))
            result.pred_boxes = Boxes(bbox)
            result.scores = score
            result.pred_classes = pred_cls
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            r = detector_postprocess(result, height, width)
            outputs = [{"instances":r}]

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results, dummy_inputs

def convert(onnx_model, dst, dummy_inputs):
    import warnings
    from torch.onnx import OperatorExportTypes
    dummy_inputs = {"image": dummy_inputs}
    # dummy_inputs = {"image": torch.randint(0, 255, [3, 562, 1000], dtype=torch.uint8, device='cuda')}
    output_names = ["bbox", "score", "pred_cls", "images_shape", "out_num"]
    dynamic_axes = {
            "image": {1: "width", 2: "height"},
        }
    onnx_model.eval()
    torch.no_grad()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(dst, 'wb') as f:
            print(f"Exporting onnx model to {dst}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=True,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                opset_version=15,
                do_constant_folding=False,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                custom_opsets={"mmdet": 15}
            )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="pmrcnn")

    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    onnx_model = OnnxModel(model)
    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    evaluator = Trainer.build_evaluator(cfg, dataset_name)
    results, dummy_inputs = inference_on_dataset(onnx_model, data_loader, evaluator)
    onnx_dst = "%s/onnx/model.onnx" % cfg.OUTPUT_DIR
    convert(onnx_model, onnx_dst, dummy_inputs)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_dst, providers=providers)
    results = inference_on_dataset(ort_session, data_loader, evaluator)
    return results


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
