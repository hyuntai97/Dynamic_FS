'''
"Dynamic Feature-Selection" Code base 

Code author: Hyuntae Kim (soodaman97@cau.ac.kr), Hyeryn Park (qkrgpfls1201@gmail.com)
----------------------------------------------------------

Selector.py

(1) LSTMCell 
    - LSTM cell 

(2) Model 
    - time series forecasting (no feature selection)
'''

import torch
import torch.nn as nn
import math

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
        self.output_dim = configs.enc_in
        self.pred_len = configs.pred_len
        self.device = configs.gpu

        self.lstm = LSTMCell(self.input_dim, self.hidden_dim, self.layer_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim*self.pred_len)
       
    def forward(self, x) :
        h0 = (torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device))
        c0 = (torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device))
        
        outs = []
        cn =  c0[0,:,:] # 32,50
        hn = h0[0,:,:]

        x = x.reshape(x.shape[0], x.shape[1], -1)   # 32,7,256
       
        for seq in range(x.size(1)) :
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            if seq == 0:
                outs = hn.unsqueeze(1)
            else:
                outs = torch.cat((outs, hn.unsqueeze(1)), axis=1)    # 224,50 (32*7, 50)

        out = outs.reshape(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.reshape(-1, self.pred_len, self.output_dim)

        return out