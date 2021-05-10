# create lstm
import torch


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size*sequence_length, num_classes)
        self.device = device

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, (h0, c0))
        out = out.reshape(out.reshape[0], -1)
        out = self.fc(out)
        return out
