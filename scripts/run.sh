ARCH="resnet50"
WIDTH="64-128-256-512-64"

INIT_LR="1.2"
LR_SCHEDULE="cosine"
LR_DECAY_EPOCH="30-70-90"
GLOBAL_BATCH_SIZE="2048"
LOCAL_BATCH_SIZE="256"
EPOCHS="500"
WARMUP="10"
AMP=""
SEED="914"

DENSITY="0.1"
RIGL_ITER="0.8"
DISTRIBUTION="uniform"


LOAD_CKPT="xxxxx"
SAVE_DIR="./checkpoints/rigl/resnet50/ep${EPOCHS}_warmup_${WARMUP}_bs_${GLOBAL_BATCH_SIZE}_lr_${INIT_LR}_with_trick/sp0.9_${DISTRIBUTION}_cosine/"


EXTRA_ARG='' #'--no-checkpoints --short-train'

cd ../

python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr ${INIT_LR} --lr-decay-epochs ${LR_DECAY_EPOCH} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.9 --wd 1e-4 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2  --widths=${WIDTH} --checkpoint-dir=${SAVE_DIR} --log-filename=${SAVE_DIR}log.log ${EXTRA_ARG} --dense-allocation=${DENSITY} --delta=100 --alpha=0.3 --T-end-percent=${RIGL_ITER} --sp-distribution=${DISTRIBUTION}

