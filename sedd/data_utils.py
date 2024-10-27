import re 
import numpy as np
from pathlib import Path
from itertools import chain 
import sys 
import os 
from transformers import GPT2TokenizerFast 
from datasets import load_dataset 
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(home_dir)
from torch.utils.data import TensorDataset
import torch 
import pickle 
import pandas as pd 
import csv 
from torch.utils.data import DataLoader, Dataset, Subset
import json 

class ProofDataset:
    def __init__(self, json_file='/home/justin/Desktop/Code/diffusion-theorem-provers/data/math.jsonl'):
        self.json_file = json_file 
        self.comb_strs = []
        self.load_seqs()
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.EOS = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.tokens = self.tokenizer([text + self.tokenizer.eos_token for text in self.comb_strs], padding=True, truncation=True,\
                                      max_length=1024, return_attention_mask=False, return_tensors='pt')

    def load_seqs(self):
        jsonObj = pd.read_json(path_or_buf=self.json_file, lines=True)
        for i in range(len(jsonObj)):
            name = jsonObj['name'][i]
            statement = jsonObj['statement'][i]
            proof = jsonObj['proof'][i]
            new_seq = f'Name: {name}\n Statement: {statement}\n Proof: {proof}'
            self.comb_strs.append(new_seq)

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx]        


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

    
def get_dataset(mode:str):
    '''
    Loads numpy dataset and returns a pytorch dataset

    dataset_path: (str) Should be filepath of numpy file that is of shape [num_samples, sample_length]
    mode: (str) Determine whether to return 'train', 'validation', or 'test' data 
    with_initial_puzzles: (bool) If true returns initial puzzle dataset, solution dataset. Else returns solution dataset 
    num_training_samples: (int | None) If given, limits the training set to the first num samples 

    ''' 
    dataset = ProofDataset()
    # Calculate the split index (80% mark)
    total_samples = len(dataset)
    split_idx = int(0.9 * total_samples)
    
    if mode.lower() == 'train':
        # For training, take the first 80%
        return Subset(dataset, range(split_idx))
        
    elif mode.lower() == 'val':
        # For validation, take the last 20%
        return Subset(dataset, range(split_idx, total_samples))

    else:
        raise ValueError("Incorrect mode")


def get_dataloaders(config):
    if config['training']['batch_size'] % (config['ngpus'] * config['training']['accum']) != 0:
            raise ValueError(f"Train Batch Size {config['training']['batch_size']} is not divisible by {config['ngpus']} gpus with accumulation {config['training']['accum']}.")
    if config['eval']['batch_size'] % (config['ngpus'] * config['training']['accum']) != 0:
        raise ValueError(f"Eval Batch Size for {config['eval']['batch_size']} is not divisible by {config['ngpus']} gpus with accumulation {config['training']['accum']}.")

    train_set = get_dataset("train")
    valid_set = get_dataset("val")

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config['training']['batch_size'] // (config['ngpus'] * config['training']['accum']),
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config['training']['batch_size'] // (config['ngpus'] * config['training']['accum']),
        num_workers=4,
        pin_memory=True,
        shuffle=True
    ))
    return train_loader, valid_loader


if __name__ == "__main__":
    ds = ProofDataset()
    breakpoint()
    print("h")
