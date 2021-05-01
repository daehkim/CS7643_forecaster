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

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        if self.model == 'LSTM':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)
        elif self.model == 'GRU':
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.device = device

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        if self.model == 'LSTM':
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        elif self.model == 'GRU':
            out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = out[:, -1, :]
        out = self.fc(out)
        # out.size() --> 100, 10
        return out
