import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from networks.networks import ActorNetwork,CriticNetwork
from utils.data_utils import ReplayBuffer_offline
from utils.noise import OUActionNoise,GausActionNoise
from agents.base import Agent_base
import minari
import copy

class TD3BC(Agent_base):
    def __init__(self,cfg,n_states,n_actions,max_action,dataset):

        self.n_states             = n_states
        self.n_actions            = n_actions
        self.max_action           = max_action

        self.device               = cfg['device']
        self.hid_layers           = cfg['hid_layers']
        self.batch_size           = cfg['batch_size']
        self.lr_actor             = cfg['lr_actor']
        self.lr_critic            = cfg['lr_critic']
        self.tau                  = cfg['tau']
        self.gamma                = cfg['gamma']
        self.actor_final_init     = cfg['actor_final_init']
        self.critic_final_init    = cfg['critic_final_init']


        self.mean_act_env         = cfg['mean_act_env']
        self.std_act_env          = cfg['std_act_env']
        self.mean_act             = cfg['mean_act']
        self.std_act              = cfg['std_act']
        self.c_act_env            = cfg['c_act_env']
        self.c_act                = cfg['c_act']
        self.normalize_states     = cfg['normalize_states']
        self.alpha                = cfg['alpha']
        self.d                    = cfg['d']

        self.param_init           = cfg['param_init']
        self.norm_method          = cfg['norm_method']

        self.actor_path           = cfg['actor_path']
        self.target_actor_path    = cfg['target_actor_path']
        self.critic_path          = cfg['critic_path']
        self.target_critic_path   = cfg['target_critic_path']

        self.replay_buffer   = ReplayBuffer_offline(self.n_states,self.n_actions)
        self.replay_buffer.convert_Minari(dataset)

        if self.normalize_states:
            self.mean_bc, self.std_bc = self.replay_buffer.normalize_states()
        else:
            self.mean_bc, self.std_bc = 0, 1

        self.noise_act_env  = GausActionNoise( mean       = self.mean_act_env,
                                                std       = self.std_act_env,
                                               dim        = self.n_actions)

        self.noise_act      = GausActionNoise( mean       = self.mean_act,
                                                std       = self.std_act,
                                               dim        = self.n_actions,
                                               batch_size = self.batch_size)

        self.actor          = ActorNetwork( input_dims         = self.n_states,
                                            hid_layers         =self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.actor_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.critic        = CriticNetwork( input_dims         = self.n_states,
                                            hid_layers         = self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.critic_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.critic2       = CriticNetwork( input_dims         = self.n_states,
                                            hid_layers         = self.hid_layers,
                                            n_actions          = self.n_actions,
                                            final_init_bound   = self.critic_final_init,
                                            param_init         = self.param_init,
                                            norm_method        = self.norm_method
                                            ).to(self.device)

        self.target_actor   = copy.deepcopy(self.actor)
        self.target_critic  = copy.deepcopy(self.critic)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.target_actor.set_eval()
        self.target_critic.set_eval()
        self.target_critic2.set_eval()

        self.actor_optimizer   = optim.AdamW(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer  = optim.AdamW(self.critic.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(),lr=self.lr_critic)

    @torch.no_grad()
    def choose_action(self, state, c = None):
        self.actor.eval()
        state    = torch.tensor(state, dtype=torch.float).to(self.device)
        pi       = self.actor(state)
        epsilon  = self.noise_act_env(c=c)
        pi_prime = pi + epsilon.to(self.device)
        self.actor.train()
        return pi_prime.cpu().numpy()

    def store_transition(self, state, action, reward, state_next, done):
        pass # data given for offline RL

    def learn(self,global_step):

        state, action, state_next, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            action_next        = self.target_actor(state_next)
            action_next       += self.noise_act(c=self.c_act).to(self.device)
            action_next        = action_next.clamp(-self.max_action,self.max_action)

            critic_value_next  = self.target_critic(state_next, action_next)
            critic_value_next2 = self.target_critic2(state_next, action_next)
            min_Q              = torch.min(critic_value_next,critic_value_next2)
            target_value       = reward + self.gamma * min_Q *not_done

        self.critic.set_train()
        critic_value   = self.critic(state, action)
        critic_loss    = F.mse_loss(critic_value,target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        critic_value2   = self.critic2(state, action)
        critic_loss2    = F.mse_loss(critic_value2,target_value)
        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        if global_step % self.d==0:

            self.critic.set_eval()
            pi          = self.actor(state)
            Q           = self.critic(state, pi)
            # vanilla implementation
            policy_loss =  -torch.mean(Q) + self.alpha*F.mse_loss(pi,action)

            #original paper from Google Brain
            # lmbda       = self.alpha / Q.abs().mean().detach()
            # policy_loss =  F.mse_loss(pi,action) - lmbda*torch.mean(Q)

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.update_target_networks()

    def update_target_networks(self):

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



