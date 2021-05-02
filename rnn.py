import torch.nn as nn
import torch


# Here we define our model as a class
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device, dropout=0.2,
                 model='LSTM'):
        super(RNN, self).__init__()

        # Initialize the variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.model = model

        self.hn = None
        self.cn = None

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        if self.model == 'LSTM' or 'GARCH+LSTM':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)
        elif self.model == 'GRU' or 'GARCH+GRU':
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.device = device

    def forward(self, x):
        if self.hn is None or self.hn.shape[1] != x.shape[0]:
            self.hn = torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_dim).requires_grad_().to(self.device)
            self.cn = torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_dim).requires_grad_().to(self.device)

        if self.model == 'LSTM' or 'GARCH+LSTM':
            out, (self.hn, self.cn) = self.rnn(x, (self.hn.detach(), self.cn.detach()))
        elif self.model == 'GRU' or 'GARCH+GRU':
            out, self.hn = self.rnn(x, self.hn)

        out = out[:, -1, :]
        out = self.fc(out)

        return out
