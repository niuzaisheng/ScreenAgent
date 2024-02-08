# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from utils.models import FineTuneTrainCogAgentModel
from sat.training.model_io import save_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="base", help='version to interact with')
    parser.add_argument("--from-pretrained", type=str, default="checkpoints/merged_lora", help='pretrained ckpt')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--rank", type=int, default=0)

    parser = FineTuneTrainCogAgentModel.add_model_specific_args(parser)
    args = parser.parse_args()
    rank = args.rank

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print("rank: {}, world_size: {}".format(rank, world_size))

    # load model
    model, model_args = FineTuneTrainCogAgentModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), url='local', overwrite_args={'model_parallel_size': 1})
    model = model.eval()
    model_args.save = './checkpoints/merged_model_{}'.format(model_args.eva_args["image_size"][0])
    save_checkpoint(1, model, None, None, model_args)

if __name__ == "__main__":
    main()
