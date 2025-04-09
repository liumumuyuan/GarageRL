import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from networks.networks import ActorNetwork,CriticNetwork
from utils.data_utils import ReplayBuffer
from utils.noise import OUActionNoise
from agents.base import Agent_base

class DDPG(Agent_base):
    def __init__(self,cfg,n_states,n_actions,max_action):

        self.n_states             = n_states
        self.n_actions            = n_actions
        self.max_action           = max_action

        self.device               = cfg['device']
        self.hid_layers           = cfg['hid_layers']
        self.batch_size           = cfg['batch_size']
        self.max_buffer_len       = cfg['max_buffer_len']

        self.lr_actor             = cfg['lr_actor']
        self.lr_critic            = cfg['lr_critic']
        self.tau                  = cfg['tau']
        self.gamma                = cfg['gamma']
        self.actor_final_init     = cfg['actor_final_init']
        self.critic_final_init     = cfg['critic_final_init']

        self.param_init           = cfg['param_init']
        self.norm_method          = cfg['norm_method']
        self.actor_path           = cfg['actor_path']
        self.target_actor_path    = cfg['target_actor_path']
        self.critic_path          = cfg['critic_path']
        self.target_critic_path   = cfg['target_critic_path']
        self.replay_items         = cfg['replay_items']

        self.replay_buffer  = ReplayBuffer(self.replay_items, self.max_buffer_len)
        self.noise          = OUActionNoise(pi=np.zeros(self.n_actions))

        self.actor          = ActorNetwork( input_dims         = self.n_states,
                                            hid_layers         =self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.actor_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.target_actor   = ActorNetwork( input_dims         = self.n_states,
                                            hid_layers         =self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.actor_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.critic        = CriticNetwork( input_dims         = self.n_states,
                                            hid_layers         =self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.critic_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.target_critic = CriticNetwork( input_dims         = self.n_states,
                                            hid_layers         =self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.critic_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.target_actor.load_state_dict( self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.target_actor.set_eval()
        self.target_critic.set_eval()

        self.actor_optimizer  = optim.AdamW(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(),lr=self.lr_critic)

    @torch.no_grad()
    def choose_action(self, state,c=None):
        self.actor.eval()
        state    = torch.tensor(state, dtype=torch.float).to(self.device)
        pi       = self.actor(state)
        pi_prime = pi + torch.tensor(self.noise(), dtype=torch.float).to(self.device)
        self.actor.train()
        return pi_prime.cpu().numpy()

    def store_transition(self, state, action, reward, state_next, done):
        data_list =[state,action,reward,state_next,done]
        step_data = {}
        assert len(self.replay_items) == len(data_list), "Replay item mismatch!"
        for key, value in zip(self.replay_items, data_list):
            step_data[key] = value
        return self.replay_buffer.store_transition(**step_data)

    def learn(self):
        memo       = self.replay_buffer.sample_buffer(self.batch_size)

        state      = torch.tensor(np.array(memo['state']),
                                  dtype=torch.float).to(self.device)
        action     = torch.tensor(np.array(memo['action']),
                                  dtype=torch.float).to(self.device)
        reward     = torch.tensor(np.array(memo['reward']),
                                  dtype=torch.float).unsqueeze(-1).to(self.device)
        state_next = torch.tensor(np.array(memo['state_next']),
                                  dtype=torch.float).to(self.device)
        done       = torch.tensor(np.array(memo['done']),
                                  dtype=torch.float).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            action_next       = self.target_actor(state_next)
            critic_value_next = self.target_critic(state_next, action_next)
            target_value      = reward + self.gamma * critic_value_next *(1- done)

        self.critic.set_train()
        critic_value   = self.critic(state, action)
        critic_loss    = F.mse_loss(critic_value,target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.set_eval()
        pi          =  self.actor(state)
        policy_loss = -self.critic(state, pi)
        policy_loss = torch.mean(policy_loss)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()

    def update_target_networks(self):

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
