export PYTHONPATH=$PYTHONPATH:/home/qianhao/PMR-CNN
python3 tool/pytorch2onnx.py --num-gpus 1 --config-file configs/PMMv2/finetune_R_101.yaml --eval-only MODEL.WEIGHTS ./output/pmmv2/R_101/model_final.pth
