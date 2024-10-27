import torch
from load_model import load_model_local
import argparse 
import sampling 
import data_utils as data 
import numpy as np 
from torch.utils.data import Subset
import os 
home_dir = os.path.dirname(os.path.dirname(__file__))
import pickle 
from torch.utils.data import Subset
from utils import isValidSudoku
from transformers import GPT2TokenizerFast



def main(args):
    device = torch.device('cuda')
    model, graph, noise = load_model_local('./',device, args.model_path, args.checkpoint_num)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    prefix_ids = tokenizer(args.prefix)['input_ids']
    input_ids = prefix_ids 
    input_locs = list(range(len(prefix_ids)))
    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.num_to_eval, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.num_to_eval, args.seq_len), 'analytic', args.steps, device=device, proj_fun=proj_fun)

    samples = proj_fun(sampling_fn(model))
        
    file_dir = os.path.join(args.model_path, 'evaluate')
    file_path = os.path.join(file_dir, 'evaluation.txt')
    if not os.path.exists(file_dir): os.makedirs(file_dir, exist_ok=True)

    with open(file_path, 'w+') as file:
        text_samples = tokenizer.batch_decode(samples)
        for i in text_samples:
            print(i)
            print("=================================================")
            file.write(i)
            file.write("=================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='experiments/10-26-2024-18:40')
    parser.add_argument("--prefix", type=str, default="Name: ")
    parser.add_argument("--checkpoint_num", type=int, required=True)
    parser.add_argument("--num_to_eval", type=int, default=16) #number of puzzles to evaluate
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=1024)
    args = parser.parse_args()
    main(args)


