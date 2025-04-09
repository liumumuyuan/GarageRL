import gymnasium as gym
import numpy as np
import time
import random
import torch
from agents.td3 import TD3
from config.config_load import load_config

#############################################################################
##### setting

algo_name        = 'TD3'
config_file_path = 'experiments/td3/td3_lunar_landing_config.yaml'
env_name         = 'LunarLanderContinuous-v3'
render_mode      = 'human'
max_n_episode    = 100000

# render_mode= None
##############################################################################

env        = gym.make(env_name, render_mode=render_mode)
n_states   = env.observation_space.shape[0]
n_actions  = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

cfg        = load_config(algo_name,config_file_path)
use_seed   = cfg['use_seed']
seed       = cfg['seed']
batch_size = cfg['batch_size']
device     = torch.device(cfg['device'])
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

agent = TD3(cfg,n_states,n_actions,max_action)

score_history = []
for i in range(max_n_episode):
    done = False
    score = 0
    state, infos = env.reset()

    #state = observation; partially observable MDP treated as fully observable
    while not done:
        act = agent.choose_action(state)
        new_state, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        agent.store_transition(state, act, reward, new_state, int(done))

        if len(agent.replay_buffer) >= batch_size:
            agent.learn(i)
        score += reward
        state = new_state

    score_history.append(score)
    print(f"episode {i}, score {score:.2f}, 100 game average {np.mean(score_history[-100:]):.2f}")
