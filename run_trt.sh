export LD_LIBRARY_PATH=/home/qianhao/miniconda3/envs/trt/lib/:/home/qianhao/miniconda3/envs/trt/lib/python3.8/site-packages/torch/lib/:/home/qianhao/TensorRT/lib/:/raid/qianhao/tensorrt/TensorRT-8.6.1.6/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=$PYTHONPATH:/home/qianhao/PMR-CNN
# ./tools/dist_test.sh configs/zsd/65_15/test/zsd/zsd_TCB_test.py pth/COCO_65_15.pth 4 --json_out results/zsd_65_15.json
# python tools/onnx2trt.py onnx/COCO_65_15.onnx trt/COCO_65_15.trt
python3 tool/trt_test.py --num-gpus 1 --config-file configs/PMMv2/finetune_R_101.yaml --eval-only
