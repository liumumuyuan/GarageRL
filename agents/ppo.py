import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base import Agent_base
from torch.distributions.categorical import Categorical
from utils.data_utils  import PPOCollection
from networks.networks import ActorNetworkPPO,CriticNetworkPPO
class PPO(Agent_base):
    def __init__(self, cfg,n_states,n_actions):

        self.n_states             = n_states
        self.n_actions            = n_actions

        self.device               = cfg['device']
        self.hid_layers           = cfg['hid_layers']
        self.epsilon              = cfg['epsilon']
        self.batch_size           = cfg['batch_size']
        self.max_buffer_len       = cfg['max_buffer_len']
        self.n_epochs             = cfg['n_epochs']
        self.buffer_minsize       = cfg['buffer_minsize']
        self.critic_lr            = cfg['critic_lr']
        self.actor_lr             = cfg['actor_lr']

        self.gamma                = cfg['gamma']
        self.policy_clip          = cfg['policy_clip']
        self.gae_lambda           = cfg['gae_lambda']
        self.lmbda                = cfg['lmbda']

        self.collection     = PPOCollection()
        self.collection_len = 0

        self.actor            = ActorNetworkPPO( self.n_states,self.hid_layers,self.n_actions).to(self.device)
        self.critic           = CriticNetworkPPO(self.n_states,self.hid_layers).to(self.device)
        self.actor_optimizer  = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(),lr=self.critic_lr)

    def collect_trajectory(self, state,next_state, action, probs, value,next_value, reward, done):
        self.collection.collect_trajectory(state,next_state, action, probs, value,next_value, reward, done)

    def process_trajectory(self):
        self.collection.process_trajectory(self.gamma,self.gae_lambda)

    def trajectory_add(self):
        self.collection_len=self.collection.trajectory_add()

    def collected_data(self):
        self.batches=self.collection.collected_data()

    def choose_action(self, state):
        state  = torch.tensor(state.tolist(), dtype=torch.float).to(self.device)
        dist   = self.actor(state)
        action = dist.sample()
        probs  = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        return action, probs

    def state_value(self,state):
        state = torch.tensor(state.tolist(), dtype=torch.float).to(self.device)
        value = self.critic(state)
        value = torch.squeeze(value).item()
        return value

    def learn(self):

        state_arr, action_arr, old_prob_arr, values_arr, reward_arr, dones_arr,advantage_arr = self.collection.collected_data()

        state_arr     = torch.tensor(np.array(state_arr),dtype=torch.float).to(self.device)
        action_arr    = torch.tensor(np.array(action_arr),dtype=torch.float).to(self.device)
        old_prob_arr  = torch.tensor(np.array(old_prob_arr),dtype=torch.float).to(self.device)
        values_arr    = torch.tensor(np.array(values_arr),dtype=torch.float).to(self.device)
        advantage_arr = torch.tensor(np.array(advantage_arr),dtype=torch.float).to(self.device)

        n_data=len(state_arr)
        n_batches=n_data//self.batch_size

        for _ in range(self.n_epochs):
            indices = np.arange(n_data, dtype=np.int64)
            np.random.shuffle(indices)
            indices = [indices[i*self.batch_size:i*self.batch_size+self.batch_size] for i in range(n_batches)]

            for batch_indices in indices:

                states    = state_arr[batch_indices]
                actions   = action_arr[batch_indices]
                old_probs = old_prob_arr[batch_indices]
                values    = values_arr[batch_indices]
                advantage = advantage_arr[batch_indices]
                dist      = self.actor(states)

                critic_value   = self.critic(states).squeeze()
                new_probs      = dist.log_prob(actions)
                prob_ratio     = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                1 + self.policy_clip) * advantage
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage + values
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.lmbda * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.collection.clear_collection()
