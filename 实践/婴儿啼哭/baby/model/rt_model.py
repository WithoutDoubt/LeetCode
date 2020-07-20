import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os

# from RTransformer import RTransformer
sys.path.append('/home/ubuntu/yww/IRMLSTM')
# from model.RZTransformer import RTransformer 
from model.RTransformer import RTransformer 
# from model.RTransformer import RTransformer 
# from rezero.transformer import RZTXEncoderLayer
class RT(nn.Module):
    def __init__(self, 
                 input_size=161, 
                 d_model=320,
                 output_size=161,
                 h=4, 
                 rnn_type="GRU",    # 居然是 GRU
                 ksize=6,           # ksize = 3
                 n=1, 
                 n_level=3,
                 dropout=0.1, 
                 emb_dropout=0.1):
        super(RT, self).__init__()                                              # ksize = 3
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)
        self.sig = nn.Sigmoid()
    
    def forward(self,x):
        x = self.encoder(x)
        o = self.rt(x)
        o = self.linear(o)

        return self.sig(o)

if __name__ == '__main__':
    devices = torch.device("cuda")
    ipt = torch.rand(1, 355, 161).cuda()

    print(ipt.device)
    rt = RT().to(devices)
    opt = rt(ipt)
    # print(rt.device)

    print(opt.shape)