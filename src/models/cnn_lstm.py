import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size = 64,
            hidden_size= 64,
            batch_first=True
        )

        self.fc  = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)

        return x