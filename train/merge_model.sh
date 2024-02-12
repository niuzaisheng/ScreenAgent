#! /bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=60003
export WORLD_SIZE=4
export LOCAL_WORLD_SIZE=4

function run_cmd() { 
    python merge_model.py \
        --version chat \
        --from-pretrained ./checkpoints/cogagent-chat-ScreenAgent-<timestamp> \
        --bf16 \
        --rank $1
}

run_cmd 0 &
run_cmd 1 &
run_cmd 2 &
run_cmd 3 &
wait