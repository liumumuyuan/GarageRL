import torch
import gymnasium as gym
import numpy as np
import time
import random
from agents.ppo import PPO
from config.config_load import load_config

#############################################################################
##### setting
algo_name        = 'PPO'
config_file_path = 'experiments/ppo/ppo_cartpole_config.yaml'
env_name         = 'CartPole-v1'
render_mode      = 'human'
max_n_episode    = 100000
N                = 20  #close to frequence of learning
eval_freq        = 10
# render_mode= None
##############################################################################
env            = gym.make(env_name,render_mode = render_mode)
n_states       = env.observation_space.shape[0]
n_actions      = env.action_space.n

cfg            = load_config(algo_name,config_file_path)
use_seed       = cfg['use_seed']
seed           = cfg['seed']
buffer_minsize = cfg['buffer_minsize']
device         = torch.device(cfg['device'])
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

agent      = PPO(cfg,n_states,n_actions)
avg_score  = 0.
best_score = -np.inf
score_history = []

for episode in range(max_n_episode):
    state, _  = env.reset()
    done      = False
    score     = 0
    n_steps_in_episode = 0
    while not done:
        action, prob                   = agent.choose_action(state)
        state_next, reward, done, _, _ = env.step(action)
        value=agent.state_value(state)
        if done:
            next_value = 0
        else:
            next_value = agent.state_value(state_next)

        score     += reward
        agent.collect_trajectory(state,state_next,action,prob,value,next_value,reward,done)
        state      = state_next

        n_steps_in_episode+=1
        if n_steps_in_episode%N==0:
            agent.process_trajectory()
            agent.trajectory_add()
            if  agent.collection_len > buffer_minsize :
                agent.learn()
    #Advantage can be computed every N steps or after done==True.
    agent.process_trajectory()
    agent.trajectory_add()
    if agent.collection_len > buffer_minsize:
        agent.learn()

    score_history.append(score)
    avg_score = np.mean(score_history[-eval_freq:])

    if avg_score > best_score:
        best_score = avg_score

    if episode%eval_freq==0:
        print(f'episode {episode}, score {score:.1f}, average score {avg_score:.1f}')
