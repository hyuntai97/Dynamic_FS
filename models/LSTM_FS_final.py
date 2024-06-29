'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

Selector.py

(1) LSTMCell 
    - LSTM cell 

(2) Model 
    - dynamic feature selection for time series forecasting 
'''

import torch
import torch.nn as nn
import math
from layers.Selector import FeatureSelector

class LSTMCell_add(nn.Module) :
    def __init__(self, input, hid, layer):
        super(LSTMCell_add, self).__init__()

        self.input_size = input
        self.hidden_size = hid
        self.x2h = nn.Linear(self.input_size, 4*self.hidden_size, bias=True)
        self.h2h = nn.Linear(self.hidden_size, 4*self.hidden_size, bias=True)
        self.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def reset_parameters(self) :
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters() :
            w.data.uniform_(-std, std)
            
    def forward(self, x, hidden) :
        hx, cx = hidden

        hx = hx.unsqueeze(1).repeat(1, x.size(1), 1)
        
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.mean(axis=1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = self.sigmoid(ingate) 
        forgetgate = self.sigmoid(forgetgate) 
        cellgate = self.tanh(cellgate) 
        outgate = self.sigmoid(outgate) 
        
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, self.tanh(cy))
        
        return (hy, cy)
    
class LSTMCell(nn.Module) :
    def __init__(self, input, hid, layer):
        super(LSTMCell, self).__init__()

        self.input_size = input
        self.hidden_size = hid
        self.x2h = nn.Linear(self.input_size, 4*self.hidden_size, bias=True)
        self.h2h = nn.Linear(self.hidden_size, 4*self.hidden_size, bias=True)
        self.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def reset_parameters(self) :
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters() :
            w.data.uniform_(-std, std)
            
    def forward(self, x, hidden) :
        hx, cx = hidden
        x = x.reshape(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = self.sigmoid(ingate) 
        forgetgate = self.sigmoid(forgetgate) 
        cellgate = self.tanh(cellgate) 
        outgate = self.sigmoid(outgate) 
        
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, self.tanh(cy))
        
        return (hy, cy)

class Model(nn.Module) :
    def __init__(self, configs) :
        super(Model, self).__init__()

        self.input_dim = configs.enc_in
        self.hidden_dim = configs.hid_dim
        self.layer_dim = configs.num_layers
        self.device = configs.gpu
        self.batch_size = configs.batch_size
        self.s_lstm = LSTMCell(self.input_dim, self.hidden_dim, self.layer_dim)
        self.p_lstm = LSTMCell(self.input_dim, self.hidden_dim, self.layer_dim)
        self.s_fc = nn.Linear(self.hidden_dim, self.input_dim)         # mask making
        self.p_fc = nn.Linear(self.hidden_dim, self.input_dim*2)        # next step predict

        # self.p_fc = nn.Sequential(
        #         nn.Linear(self.hidden_dim, 150),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(150, self.input_dim*2)
        #     )     # next step predict

        self.fs = FeatureSelector(configs).to(self.device)

    def forward(self, x) :

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device) # 3,32,50
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
           
        cn = c0[0,:,:] # 32,50
        hn = h0[0,:,:]
        cn2 = c0[0,:,:] # 32,50
        hn2 = h0[0,:,:]
       
        for seq in range(x.size(1)-1) :

            if seq == 0:
                hn, cn = self.s_lstm(1 * x[:, seq, :],(hn,cn))
            else:   
                hn, cn = self.s_lstm(mask_n * x[:, seq, :] + (1-mask_n) * xhat0, (hn,cn))    # (32,256)
                
            mask_n, reg, gate, z = self.fs(hn)  # (32,256)
            mu, prob = gate
            mu = torch.tensor(mu)
            prob = torch.tensor(prob)

            hn2, cn2 = self.p_lstm(mask_n * x[:, seq+1, :], (hn2, cn2))
            xhat = self.p_fc(hn2).reshape(-1, 2, self.input_dim)
            xhat0 = xhat[:,0,:]
            xhat1 = xhat[:,1,:]

            if seq == 0:
                s_outs = mask_n.unsqueeze(1)
                p_outs_0 = xhat0.unsqueeze(1)
                p_outs = xhat1.unsqueeze(1)
                m_reg = reg.unsqueeze(1)
                mus = mu.unsqueeze(1)
                probs = prob.unsqueeze(1)
                zs = z.unsqueeze(1)
            else:
                s_outs = torch.cat((s_outs, mask_n.unsqueeze(1)), axis=1)    # 32,8,256
                p_outs_0 = torch.cat((p_outs_0, xhat0.unsqueeze(1)), axis=1)    # 32,8,256
                p_outs = torch.cat((p_outs, xhat1.unsqueeze(1)), axis=1)    # 32,8,256
                m_reg = torch.cat((m_reg, reg.unsqueeze(1)), axis=1)
                mus = torch.cat((mus, mu.unsqueeze(1)), axis=1)
                probs = torch.cat((probs, prob.unsqueeze(1)), axis=1)
                zs = torch.cat((zs, z.unsqueeze(1)), axis=1)

        return s_outs, p_outs_0, p_outs, m_reg, mus, probs, zs