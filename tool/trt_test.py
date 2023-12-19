import argparse
import os
import os.path as osp

import mmcv
import torch
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
from tool import engine as engine_utils
from tool import common
import pycuda.driver as cuda
import numpy as np
from functools import reduce
from typing import List, Union
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators
from detectron2.utils.comm import get_world_size
import logging
import time, datetime
from collections import abc
from pmrcnn.data.build import build_detection_test_loader
from pmrcnn.config import get_cfg
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import Instances, Boxes
from detectron2.modeling.postprocessing import detector_postprocess

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger


class Trainer(DefaultTrainer):

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
    logger = logging.getLogger("pmrcnn")
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

    start_data_time = time.perf_counter()
    for idx, inputs in enumerate(data_loader):
        total_data_time += time.perf_counter() - start_data_time
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
        start_compute_time = time.perf_counter()
        img = inputs[0]['image']
        width = inputs[0]['width']
        height = inputs[0]['height']

        dict_input = {"image": img.cpu().numpy()}
        bbox, score, pred_cls, image_shape = model.infer(dict_input)
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
        # iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        # data_seconds_per_iter = total_data_time / iters_after_start
        # compute_seconds_per_iter = total_compute_time / iters_after_start
        # eval_seconds_per_iter = total_eval_time / iters_after_start
        # total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
        # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
        #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
        #     log_every_n_seconds(
        #         logging.INFO,
        #         (
        #             f"Inference done {idx + 1}/{total}. "
        #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
        #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
        #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
        #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
        #             f"ETA={eta}"
        #         ),
        #         n=5,
        #     )
        start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {:.1f} s ({:.1f} Hz per device, on {} devices)".format(
            total_time, (total - num_warmup) / total_time , num_devices
        )
    )
    # total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} Hz per device, on {} devices)".format(
    #         total_compute_time_str, (total - num_warmup) / total_compute_time, num_devices
    #     )
    # )
    results = evaluator.evaluate()

class TrtModel(object):
    def __init__(self, engine_path, dynamic_shape={}, dynamic_shape_value={}) -> None:
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.rt = trt.Runtime(TRT_LOGGER)
        self.engine = None
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        if not os.path.exists(engine_path):
            raise Exception('tensorRT engine file not exist')
        self.engine = engine_utils.load_engine(self.rt, engine_path)
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = \
            common.allocate_buffers(self.engine, self.ctx, dynamic_shape, dynamic_shape_value)

    def infer(self, dict_input):
        for i, binding in enumerate(self.engine):
            if i == 0:
                input_arr = dict_input[binding]
                self.inputs[i].host = input_arr
                #actual_shape = list(input_arr.shape)
                #self.ctx.set_input_shape(binding, actual_shape)

        common.do_inference(self.ctx, self.bindings, self.inputs, self.outputs, self.stream)
        res = []
        num_outputs = self.outputs[5].host.item()
        for i, data in self.outputs.items():
            actual_shape = [dim for dim in self.ctx.get_tensor_shape(self.engine[i])]
            if i in (1,2,3):
                actual_shape[0] = num_outputs
            if i <= 4:
                dsize = reduce(lambda x, y: x*y, actual_shape)
                out = data.host[:dsize].reshape(actual_shape)
                res.append(out)
        return res

    def __del__(self):
        self.cfx.pop()

# def trt_test(trt_model, data_loader):
#     test_inputs = {}
#     device_id = 0
#     dataset = data_loader.dataset
#     warmup = {}
#     for data in data_loader:
#         img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
#         warmup['img'] = img.numpy()
#         warmup['img_shape'] = np.array(img_meta['img_shape'], dtype=np.int32)
#         warmup['scale_factor'] = np.array(img_meta['scale_factor'], dtype = np.float32)
#         break
#     for i in range(100):
#         det_bboxes, det_labels = trt_model.infer(warmup)
#         result = bbox2result(det_bboxes, det_labels, 66)
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     results = []
#     for data in data_loader:
#         img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
#         test_inputs['img'] = img.numpy()
#         test_inputs['img_shape'] = np.array(img_meta['img_shape'], dtype=np.int32)
#         test_inputs['scale_factor'] = np.array(img_meta['scale_factor'], dtype = np.float32)
#         det_bboxes, det_labels = trt_model.infer(test_inputs)
#         result = bbox2result(det_bboxes, det_labels, 66)
#         # print(det_labels)
#         batch_size = img.size(0)
#         for _ in range(batch_size):
#             prog_bar.update()
#         results.append(result)
#     return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(cfg.OUTPUT_DIR, name="pmrcnn", configure_stdout=True)
    return cfg


def main(args):
    cfg = setup(args)
    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    evaluator = Trainer.build_evaluator(cfg, dataset_name)
    trt_engine_path = "%s/trt/model.trt" % cfg.OUTPUT_DIR
    dynamic_input = {
        "image": [3,562,1000]
    }
    trt_model = TrtModel(engine_path=trt_engine_path, dynamic_shape=dynamic_input)
    inference_on_dataset(trt_model, data_loader, evaluator)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
