#!/bin/bash

CONFIG=$1
GPUS=$2
WORKDIR=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH

shift 3
REST_ARGS="$@"

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch $REST_ARGS --work-dir $WORKDIR
