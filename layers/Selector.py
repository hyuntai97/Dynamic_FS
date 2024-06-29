'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

Selector.py

(1) Selector class 
    - actor model 

(2) FeatureSelector Class
    - generate stochastic gate for feature selection 
'''


import torch
import torch.nn as nn
import numpy as np
import math

class Selector(nn.Module):
    def __init__(self, configs):
        super(Selector, self).__init__()

        self.actor_h_dim = configs.actor_h_dim
        self.dim = configs.enc_in

        self.actor_model = nn.Sequential(
            nn.Linear(self.dim, self.actor_h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.actor_h_dim, self.dim),
        )

    def forward(self, x):
        prob = self.actor_model(x)
        return prob
    
class FeatureSelector(nn.Module):
    def __init__(self, configs):
        super(FeatureSelector, self).__init__()
        self.device = configs.gpu
        self.mu = 0.01*torch.randn(configs.enc_in, )
        self.eps = torch.randn(self.mu.size()).to(self.device) # random noise
        self.sigma = 0.5
        self.selector = Selector(configs).to(self.device)
        self.shift = 0.5
    
    def hard_sigmoid(self, x):
        return torch.clamp(x + self.shift, 0.0, 1.0)

    def forward(self, prev_x):
        self.mu = self.selector(prev_x) * 0.5
        z = self.mu + self.sigma*self.eps.normal_()
        stochastic_gate = self.hard_sigmoid(z)
        return stochastic_gate, self.regularizer(), self.get_gates(), self.hard_sigmoid(self.mu)
    
    def regularizer(self): # Gaussian CDF
        x = (self.mu + self.shift) / self.sigma
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def get_gates(self):
        return self.mu.detach().cpu().numpy(), np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy()+self.shift)) 