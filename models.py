import torch.nn as nn

# LSTM model for client
class ClientLSTM(nn.Module):
    def __init__(self, input_shape):
        super(ClientLSTM, self).__init__()
        self.lstm = nn.LSTM(input_shape[1], 100, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x

# LSTM model for server
class ServerLSTM(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(ServerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_shape, 50, batch_first=True)
        self.fc = nn.Linear(50, n_outputs)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x