
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler




class RULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take only the last timestep
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)
