#! /bin/bash

set +x

num_machines=1
NUM_GPUS_PER_WORKER=8
num_processes=8

export WORLD_SIZE=$num_processes
export MASTER_ADDR=localhost
export MASTER_PORT=60000

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

MP_SIZE=4

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="cogagent-chat"
JOB_NAME="ScreenAgent"
VERSION="chat"
experiment_name="$MODEL_TYPE-$JOB_NAME"

MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 2048 \
    --local_tokenizer lmsys/vicuna-7b-v1.5 \
    --version $VERSION"
    # --lora_rank 50 \
    # --use_lora \

# TIPS: max_length include low-resolution image sequence (which has 256 tokens) 

OPTIONS_SAT="SAT_HOME=$script_dir/saved_models"
OPTIONS_NCCL="NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

gpt_options=" \
       --experiment-name $experiment_name \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --epochs 1 \
       --resume-dataloader \
       $MODEL_ARGS \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 1000 \
       --eval-interval 5000 \
       --save "./checkpoints" \
       --strict-eval \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config deepspeed_config.json \
       --skip-init \
       --seed 42
"

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} finetune_ScreenAgent.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}
