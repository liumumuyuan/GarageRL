import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

def uni_param_init(layer,bound = None):
    if bound is None:
        bound  =  1 / np.sqrt(layer.in_features)
    torch.nn.init.uniform_(layer.weight.data, -bound, bound)
    torch.nn.init.uniform_(layer.bias.data,   -bound, bound)

class Layer(nn.Module):
    def __init__(self,
                 input_dims,
                 output_dims,
                 param_init_fn = None,
                 norm_method   = None,
                 ):
        super().__init__()
        self.fc = nn.Linear(input_dims,output_dims)
        if param_init_fn:
           param_init_fn(self.fc)

        if norm_method   == 'layer norm':
            self.norm = nn.LayerNorm(output_dims)
        elif norm_method == 'batch norm':
            self.norm = nn.BatchNorm(output_dims)
        elif norm_method is None:
            self.norm = nn.Identity()
        else:
            raise ValueError("not implemented")


    def forward(self,x):
        x = self.fc(x)
        x = self.norm(x)
        return x

class BaseNetwork(nn.Module):
    def __init__(self,
                 input_dims,
                 hid_layers,
                 n_actions         = None,
                 final_init_bound  = None,
                 param_init        = None,
                 norm_method       = None,
                 ):
        super().__init__()

        if param_init == 'uniform':
            self.param_init_fn = uni_param_init
        else:
            self.param_init_fn = None

        self.fc_input   = Layer( input_dims,
                                 hid_layers[0],
                                 self.param_init_fn ,
                                 norm_method)

        self.fc_hids    = nn.ModuleList([
                          Layer(hid_layers[i],hid_layers[i+1],
                                self.param_init_fn,norm_method)
                          for i in range(len(hid_layers)-1)
                          ])
    def forward(self,x):
        x = self.fc_input(x)
        x = F.relu(x)
        for layer in self.fc_hids:
            x = layer(x)
            x = F.relu(x)
        return x

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()

class ActorNetwork(BaseNetwork):
    def __init__(self, input_dims,
                       hid_layers,
                       n_actions,
                       final_init_bound = 0.003,
                       param_init       = 'uniform',
                       norm_method      = 'layer norm',
                       max_action       = 1.):

        super(ActorNetwork, self).__init__( input_dims,
                                            hid_layers,
                                            n_actions,
                                            final_init_bound,
                                            param_init,
                                            norm_method )
        self.max_action = max_action
        self.fc_output  = nn.Linear(hid_layers[-1],n_actions)
        if self.param_init_fn:
            self.param_init_fn(self.fc_output,final_init_bound)

    def forward(self, state):
        x =super().forward(state)
        x = self.fc_output(x)
        x = self.max_action*torch.tanh(x)
        return x

class CriticNetwork(BaseNetwork):

    def __init__(self, input_dims,
                       hid_layers,
                       n_actions,
                       final_init_bound = 0.003,
                       param_init       = 'uniform',
                       norm_method      = 'layer norm' ):

        super(CriticNetwork, self).__init__( input_dims,
                                            hid_layers,
                                            n_actions,
                                            final_init_bound,
                                            param_init,
                                            norm_method )

        self.action_value = Layer(n_actions,
                                  hid_layers[-1],
                                  self.param_init_fn,
                                  norm_method)

        self.q = nn.Linear(hid_layers[-1],1)
        if self.param_init_fn:
            self.param_init_fn(self.q,final_init_bound)

    def forward(self, state, action):
        # late merge for now, TODO: early merge for extention
        state_value = self.fc_input(state)
        state_value = F.relu(state_value)
        for i,layer in enumerate(self.fc_hids):
            state_value = layer(state_value)
            if i<len(self.fc_hids)-1:
                state_value = F.relu(state_value)

        action_value  = self.action_value(action)
        state_action_value = F.relu(state_value + action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

class ActStochas(BaseNetwork):
    def __init__(self, input_dims,
                       hid_layers,
                       n_actions,
                       final_init_bound = 0.003,
                       param_init       = 'uniform',
                       norm_method      = 'layer norm',
                       log_clamp_min    = -20,
                       log_clamp_max    = 2,
                       epsilon          = 1e-6):

        self.log_clamp_min = log_clamp_min
        self.log_clamp_max = log_clamp_max
        self.epsilon       = epsilon
        super(ActStochas, self).__init__( input_dims,
                                            hid_layers,
                                            n_actions,
                                            final_init_bound,
                                            param_init,
                                            norm_method )
        self.head_mean    = nn.Linear(hid_layers[-1],n_actions)
        self.head_log_std = nn.Linear(hid_layers[-1], n_actions)

    def forward(self, state):
        x        = super().forward(state)
        mean     = self.head_mean(x)
        log_std  = self.head_log_std(x)
        log_std  = torch.clamp(log_std,min=self.log_clamp_min,max=self.log_clamp_max)
        std = torch.exp(log_std)    # x = torch.normal(mean=mean, std=std) do NOT support gradient!
        normal   = Normal(mean, std)
        z        = normal.rsample() # rsample() is differentiable, r for reparameterization
        action   = torch.tanh(z)
        log_prob = normal.log_prob(z)
        entropy  = normal.entropy() # now entropy only for original distribution,
                                    # i.e. z instead for tanh(z)
                                    # extention meaningful?
                                    # not important, as not used in core algo

        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)

        return action,log_prob,entropy

class ActorNetworkPPO(BaseNetwork):
    def __init__(self, input_dims,
                       hid_layers,
                       n_actions,
                       final_init_bound = None,
                       param_init       = None,
                       norm_method      = None ):
        super(ActorNetworkPPO, self).__init__( input_dims,
                                               hid_layers,
                                               n_actions,
                                               final_init_bound,
                                               param_init,
                                               norm_method )
        self.ll      = nn.Linear(hid_layers[-1],n_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = super().forward(x)
        x = self.ll(x)
        dist = self.softmax(x)
        dist = Categorical(dist)
        return dist

class CriticNetworkPPO(BaseNetwork):
    def __init__(self, input_dims,
                       hid_layers,
                       n_actions        = None,
                       final_init_bound = None,
                       param_init       = None,
                       norm_method      = None ):
        super(CriticNetworkPPO, self).__init__( input_dims,
                                            hid_layers,
                                            n_actions,
                                            final_init_bound,
                                            param_init,
                                            norm_method )
        self.critic_value = nn.Linear(hid_layers[-1],1)
    def forward(self,x):
        x = super().forward(x)
        return self.critic_value(x)
