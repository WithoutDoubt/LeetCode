import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        """Construct LSTM model.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=161, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=161)
        self.activation = nn.Sigmoid()

    def forward(self, ipt):
        self.lstm.flatten_parameters()  
        o, h = self.lstm(ipt)
        o = self.linear(o)
        o = self.activation(o)
        return o


if __name__ == '__main__':
    ipt = torch.rand(1, 355, 161)
    print(ipt.device)
    opt = LSTMModel()(ipt)
    print(opt.device)
    print(opt.shape)
