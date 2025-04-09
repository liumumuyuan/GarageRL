import numpy as np
import torch

class GausActionNoise(object):
    def __init__(self,mean=0., std=1.0,dim=1,batch_size=None):
        self.mean       = mean
        self.std        = std
        self.dim        = dim
        self.batch_size = batch_size

    def __call__(self,c=None):
        if self.batch_size is None:
            means   = torch.ones([self.dim],dtype = torch.float)*self.mean
            stds    = torch.ones([self.dim],dtype = torch.float)*self.std
        else:
            means   = torch.ones([self.batch_size,self.dim],dtype = torch.float)*self.mean
            stds    = torch.ones([self.batch_size,self.dim],dtype = torch.float)*self.std
        x           = torch.normal( mean = means, std = stds)
        if c is not None:
            return torch.clip(x, min=-c, max=c)
        return x

# Ornstein-Uhlenbeck Action Noise
class OUActionNoise(object):
    def __init__(self, pi, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.pi = pi
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (self.x_prev +
             self.theta * (self.pi - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.pi.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.pi)
