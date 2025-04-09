import sys
import time
import os
import torch
import gymnasium as gym
import minari
import numpy as np
import yaml
import random
from agents.td3bc import TD3BC
from config.config_load import load_config

def eval_policy(agent,mean,std, seed, seed_offset=0, eval_episodes=1):
    dataset = minari.load_dataset(f"mujoco/hopper/medium-v0", download=True)
    eval_env = dataset.recover_environment()
    eval_env.reset(seed=seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_norm = (state - mean) / std
            action = agent.choose_action(state_norm)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

#############################################################################
##### run setting

algo_name        = 'TD3BC'
config_file_path = 'experiments/td3_bc/td3_bc_hopper_config.yaml'
render_mode      = 'human'
env_name         = "mujoco/hopper/medium-v0"
vis_env_name     = 'Hopper-v5'

max_steps        = 10000000
eval_freq        = 5000
vis_freq         = 5000
seed_offset      = 100
eval_episodes    = 10
# render_mode= None
##############################################################################

dataset    = minari.load_dataset(env_name, download=True)
env        = dataset.recover_environment()
n_states   = env.observation_space.shape[0]
n_actions  = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

cfg              = load_config(algo_name,config_file_path)
use_seed         = cfg['use_seed']
seed             = cfg['seed']
normalize_states = cfg['normalize_states']
device           = torch.device(cfg['device'])
print('Device is set to',device)

if use_seed:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() is True:
        torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

agent = TD3BC(cfg,n_states,n_actions,max_action,dataset)
if normalize_states:
    mean  = agent.mean_bc.squeeze(0)
    std   = agent.std_bc.squeeze(0)
else:
    mean  =0.
    std   =1.
score_history = []
for i in range(max_steps):

    agent.learn(i)

    if (i + 1) % eval_freq == 0:
        print(f"Time steps: {i+1}")
        score = eval_policy(agent, mean,std, seed,seed_offset,eval_episodes)
        score_history.append(score)

    if (i + 1) % vis_freq  == 0:
        vis_env  = gym.make(vis_env_name, render_mode=render_mode)
        state, _ = vis_env.reset(seed=seed+seed_offset)
        done     = False

        while not done:
            norm_state = (state - mean) / std
            action     = agent.choose_action(norm_state)
            state, reward, terminated, truncated, _ = vis_env.step(action)
            vis_env.render()
            done       = terminated or truncated
        vis_env.close()
