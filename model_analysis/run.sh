## extract one model state_dict and save to npy file
python model_analysis.py --mode extract_single --model-path ori_models/checkpoint-0.pth.tar --weight-path model_weight/checkpoint-0.npy --sp-config-file ../profiles/resnet_0.8.yaml

## extract batch models state_dict and save to npy file (need to manually modify model name and dir and rull of iteration)
python model_analysis.py --mode extract_batch --model-path ori_models/checkpoint-0.pth.tar --weight-path model_weight/checkpoint-0.npy --sp-config-file ../profiles/resnet_0.8.yaml


## check two model weight sparse mask IOU
python model_analysis.py --mode iou_single_pair --iou-path-1 model_weight/checkpoint-0.npy --iou-path-2 model_weight/checkpoint-150.npy

## check batch models weight sparse mask IOU with all combinations
python model_analysis.py --mode iou_batch --iou-path-1 model_weight/checkpoint-0.npy --iou-path-2 model_weight/checkpoint-150.npy


## check model sparsity
python model_analysis.py --mode sparsity --model-path ../checkpoints/xxxxx-xx-x-xxx-xx-x.pth.tar


## plot weight distribution of model
python model_analysis.py --mode plot --weight-path model_weight/xxx-xx-x-xx-xxx.npy --fig-path fig/xxx-xxx.png




