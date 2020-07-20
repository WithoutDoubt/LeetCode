
import torch.nn as nn
import MetaRNNs
import torch
from torch.autograd import Variable
torch.manual_seed(100)

test_rnn = MetaRNNs.MetaRNN(4, 8, 4, 4, 1)
test_input = Variable(torch.randn(1, 5, 4))
# test_h0 = Variable(torch.randn(1, 8))
test_output, test_hn = test_rnn(input)
print("------------test_input------------")
print(test_input)
print("------------test_output------------")
print(test_output)
print("--------------test_hn--------------")
print(test_hn)
