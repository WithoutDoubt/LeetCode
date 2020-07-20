import math
import torch
import torch.nn as nn
import warnings
import itertools
import numbers
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.MetaRNNs import MetaLSTM



from torch.nn import Module
from torch.nn import Parameter
from torch.nn.modules.rnn import PackedSequence
from torch.nn import init


class meta_LSTMModel(nn.Module):
    def __init__(self, input_size=161, hidden_size=1024, hyper_hidden_size=512, hyper_embedding_size=512, num_layers=2, num_classes=161, bias=True):
        super(meta_LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_gpu = True
        self.num_gpu = 1

        self.rnn = MetaLSTM(input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, num_layers=num_layers, bias=bias)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

        self.linear = nn.Linear(in_features=1024, out_features=161)
        self.activation = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
            # weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # set initial states
        # initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)) for _ in range(self.num_layers)]
        # self.rnn.flatten_parameters()
        # forward propagate RNN
        # self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        # print('out0-------')
        # print(out.shape)
        o = self.linear(out)
        o = self.activation(o)

        # out = out[:, -1, :]
        # # print('out1------')
        # # print(out.size())
        # out.view(-1, self.hidden_size)
        # # print('out2----------')
        # # print(out.size())
        # out = self.fc(out)
        # # print('out3--------')
        # # print(out.size())
        # out = out.view(-1, self.num_classes)
        # # print('out4----------')
        # # print(out.size())
        return o

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
if __name__ == '__main__':
    
    #test_lstm = MetaLSTM(4,8,4,4,1)

    # test_input = torch.randn(1, 5, 4)
    # opt,_ = test_lstm(test_input)

    # print('opt.shape = {}'.format(opt.shape))

#     x = torch.rand(355,161)
#     y = torch.rand(355,161)
    ipt = torch.rand(1,355,161)
# #    ipt = Variable(ipt)
    opt = meta_LSTMModel()(ipt)
    print(opt.shape)