"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import pdb
import os

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #logger.info("saving %s", self.config.ckpt_path)
        logger.info("saving %s", self.config.save_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.save_path + '/' + str(epoch) + '.pth')

    def _image_aug_rand_shift(self, x, y, r, t, training_option, padding_size=10):
        if training_option == 0:
            return x, y, r, t
        else:
            B, T, CHW = x.shape
            imgs = x.reshape(B*T, 4, 84, 84).numpy()

            n, c, h, w = imgs.shape

            w1 = np.random.randint(0, 2 * padding_size, n)
            h1 = np.random.randint(0, 2 * padding_size, n)

            rand_shifts = np.zeros((n, c, h, w), dtype = imgs.dtype)

            for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
                
                blank_img = np.zeros((c, h + 2 * padding_size, w + 2 * padding_size), dtype = imgs.dtype)
                blank_img[:, padding_size:padding_size + h, padding_size:padding_size + w] = img.copy()

                rnd = torch.rand((1,))
                if rnd > 0.5:
                    rand_shifts[i] = blank_img[:, h11 : h11 + h , w11 : w11 + w]
                else:
                    rand_shifts[i] = img.copy()
                
                
            
            rand_shifts = torch.from_numpy(rand_shifts)
            
            rand_shifts = rand_shifts.reshape(B, T, CHW)
            x = torch.cat([x, rand_shifts], dim=0)
            y = torch.cat([y, y], dim=0)
            r = torch.cat([r, r], dim=0)
            t = torch.cat([t, t], dim=0)
            return x, y, r, t

    def _image_aug_cutout(self, x, y, r, t, training_option, min_cut=10,max_cut=30):

        """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
        """
        
        if training_option == 0:
            return x, y, r, t
        else:
            
            B, T, CHW = x.shape
            #rand_idx = torch.randint(0, B - 1, (B // 2,))  # torch.Size([64])

            #x_og = x[rand_idx]  # B//2 x T x CHW    #torch.Size([64, 30, 28224])
            #imgs = x_og.reshape((B//2)*T, 4, 84, 84).numpy()  # (1920, 4, 84, 84)
            imgs = x.reshape(B*T, 4, 84, 84).numpy()

            n, c, h, w = imgs.shape
            w1 = np.random.randint(min_cut, max_cut, n)
            h1 = np.random.randint(min_cut, max_cut, n)
            
            cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
            for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
                cut_img = img.copy()

                rnd = torch.rand((1,))
                if rnd > 0.5:
                    cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
                #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
                cutouts[i] = cut_img
            
            cutouts = torch.from_numpy(cutouts)
            #cutouts = cutouts.reshape(B//2, T, CHW)
            #cutouts = torch.cat([x_og, cutouts], dim=0)  # torch.Size([128, 30, 28224])
            cutouts = cutouts.reshape(B, T, CHW)
            x = torch.cat([x, cutouts], dim=0)
            y = torch.cat([y, y], dim=0)
            r = torch.cat([r, r], dim=0)
            t = torch.cat([t, t], dim=0)
            return x, y, r, t


        """
        elif training_option == 1:

            B, T, CHW = x.shape  # 128 x 30 x 28224

            #imgs = x.reshape(B*T*4, 1, 84, 84).numpy()  #frame stack에서 frame마다 같은 위치에 cutout?
            imgs = x.reshape(B*T, 4, 84, 84).numpy()   # 3840 x 4 x 84 x 84

            n, c, h, w = imgs.shape
            w1 = np.random.randint(min_cut, max_cut, n)
            h1 = np.random.randint(min_cut, max_cut, n)
            
            cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
            for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
                cut_img = img.copy()

                rnd = torch.rand((1,))
                if rnd > 0.5:
                    cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
                #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
                cutouts[i] = cut_img
            
            cutouts = torch.from_numpy(cutouts)
            cutouts = cutouts.reshape(B, T, CHW)
            return cutouts
        """


        

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):


            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                

                if config.aug_type == 'cutout':
                    x, y, r, t = self._image_aug_cutout(x, y, r, t, config.training_option, 10, 30)
                elif config.aug_type == 'random_shift':
                    x, y, r, t = self._image_aug_rand_shift(x, y, r, t, config.training_option, 10)
                else:
                    raise NotImplementedError()
                
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        
        best_return = -float('inf')
        
        self.tokens = 0 # counter used for learning rate decay
        
        eval_ret = []
        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')
            
            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()
            
            self.save_checkpoint(epoch)
            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
                eval_ret.append(eval_return)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90)
                    eval_ret.append(eval_return)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150)
                    eval_ret.append(eval_return)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000)
                    eval_ret.append(eval_return)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                    eval_ret.append(eval_return)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            with open(self.config.save_path + '/eval_return.txt', "w") as file:
                file.write(str(eval_ret))

    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
