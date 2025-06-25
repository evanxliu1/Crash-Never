import torch
import torch.nn as nn

class LSTMSequenceModel(nn.Module):
    def __init__(self, input_dim=525, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Use last hidden state for classification
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
