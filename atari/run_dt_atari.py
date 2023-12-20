import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset

import os
from datetime import datetime
from omegaconf import OmegaConf
import pdb
import json

#python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 5000 --num_buffers 50 --game 'Pong' --batch_size 128 --training_option 2
#python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128 --training_option 0

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
parser.add_argument('--result_path', type=str, default='./results/')
parser.add_argument('--training_option', type=int, default=0)
parser.add_argument('--attn_loss_ratio', type=float, default=1.0)
parser.add_argument('--aug_type', type=str, default='cutout')
#parser.add_argument('--data_dir_prefix', type=str, default='/data/scyu/project/dt/decision-transformer/atari/datasets/')
args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

if args.training_option != 0:
    args.num_steps = args.num_steps // 2
    print("training_option : ", args.training_option)
    print("num_steps changed")

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)


# obss : list, len : 17002 [... (4, 84, 84) ...]
# actions : np array, (17002,)
# done_idxs : np array, (11,)     array([  676,  2196,  3685,  5043,  6868,  8874, 10423, 11611, 13357, 15174, 17002])
# rtgs : np array (17002,)
# timesteps : np array (17003,)      timesteps[676] : 676, timesteps[677] : 0


path = args.result_path + datetime.today().strftime("%Y%m%d_%H%M") + "_" + args.game + '_' + str(args.seed) + '_' + str(args.training_option) + '_' + args.aug_type
os.mkdir(path)

with open(path + '/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps),
                  training_option=args.training_option, attn_loss_ratio=args.attn_loss_ratio)
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),
                      save_path=path, training_option = args.training_option, attn_loss_ratio=args.attn_loss_ratio,
                      aug_type=args.aug_type)

trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
