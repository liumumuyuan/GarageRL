import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # Fixed typo: was "funtional"
import torch.optim as optim
import numpy as np
from collections import deque

class ReplayBuffer_offline(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
    def convert_Minari(self, dataset):
        for episode in dataset.iterate_episodes():
            obs = episode.observations
            actions = episode.actions
            next_obs = episode.observations[1:]
            rewards = episode.rewards
            dones = episode.terminations | episode.truncations

            for i in range(len(rewards)):
                self.add(obs[i], actions[i], next_obs[i], rewards[i], dones[i])

class ReplayBuffer(object):
    ### FIFO by deque, chached for efficiency
    def __init__(self,fields, max_buffer_len):
        self.fields = fields
        self.data = {}
        for k in self.fields:
            self.data[k] = deque(maxlen=max_buffer_len)

    def store_transition(self,**kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)

    def clear(self):
        for k in self.fields:
            self.data[k].clear()

    def __len__(self):
        for v in self.data.values():
            return len(v)
        return 0

    def sample_buffer(self, batch_size):
        batch = np.random.choice(len(self),batch_size)
        result ={}
        for k in self.fields:
            chached = list(self.data[k]) #caching
            result[k] =[chached[i] for i in batch]
        return result

class ReplayBuffer__(object):
    def __init__(self,fields=None):
        self.data = {}
        for key in fields:
            self.data[key] = []

    def store_transition(self,**kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)

    def clear(self):
        for k, v in self.data.items():
            self.data[k] = []

    def __len__(self):
        for v in self.data.values():
            return len(v)
        return 0

# Replay Buffer for experience replay
class ReplayBuffer_(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # Create buffers with proper shape tuples
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done  # 1 if not done, 0 if done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]      # Fixed indexing: use square brackets
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class PPOCollection:
    def __init__(self):

        self.states = []
        self.next_states = []
        self.probs = []
        self.values = []
        self.next_values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantage = []

        self.states_collection = []
        self.next_states_collection = []
        self.probs_collection = []
        self.values_collection = []
        self.next_values_collection = []
        self.actions_collection = []
        self.rewards_collection = []
        self.dones_collection = []
        self.advantage_collection = []

    def clear_trajectory(self):
        self.states = []
        self.next_states = []
        self.probs = []
        self.values = []
        self.next_values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantage = []

    def clear_collection(self):
        self.states_collection = []
        self.next_states_collection = []
        self.probs_collection = []
        self.values_collection = []
        self.next_values_collection = []
        self.actions_collection = []
        self.rewards_collection = []
        self.dones_collection = []
        self.advantage_collection = []

        self.clear_trajectory()


    def collect_trajectory(self, state,next_state, action, probs, value,next_value, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(value)
        self.next_values.append(next_value)
        self.rewards.append(reward)
        self.dones.append(done)

    def trajectory_add(self):
        if self.states == []:
            self.clear_trajectory()
            return len(self.states_collection)

        self.states_collection+=self.states#.append(self.states) #+=self.states.tolist()
        self.probs_collection+=self.probs
        self.values_collection+=self.values
        self.actions_collection+=self.actions
        self.rewards_collection+=self.rewards
        self.dones_collection+=self.dones
        self.advantage_collection+=self.advantage

        self.clear_trajectory()

        return len(self.states_collection)

    def collected_data(self):
        return (self.states_collection,
                self.actions_collection,
                self.probs_collection,
                self.values_collection,
                self.rewards_collection,
                self.dones_collection,
                self.advantage_collection)

    def process_trajectory(self, gamma, gae_lambda):
        self.advantage = [0] * len(self.rewards)
        gae = 0

        for t in reversed(range(len(self.rewards) )):
            delta = self.rewards[t] + gamma * self.next_values[t] - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            self.advantage[t] = gae

    def process_trajectory_origin(self,gamma,gae_lambda): # no GAE, no Rollout
        for t in range(len(self.rewards)):
            a_t = self.rewards[t] + gamma * self.next_values[t] - self.values[t]
            self.advantage.append(a_t)

    def process_trajectory_roll_out(self,gamma,gae_lambda): #not called
        roll_out_steps=4
        for t in range(len(self.rewards)):
            roll_out_steps_onsite=min(roll_out_steps,len(self.rewards)-t)
            return_value =self.next_values[t+roll_out_steps_onsite-1]
            for step in range(roll_out_steps_onsite)[::-1]:
                return_value = self.rewards[t+step]+gamma*return_value

            a_t = return_value-self.values[t]
            self.advantage.append(a_t)
