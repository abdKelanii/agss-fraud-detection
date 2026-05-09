"""
Fraud detection models (PyTorch).
LSTM/RNN input shape: (batch, seq_len=1, n_features)
GAN input shape: (batch, n_features) or (batch, 1, n_features) — seq dim is squeezed
"""

import torch
import torch.nn as nn


class FraudLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class FraudRNN(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class FraudGAN(nn.Module):
    """
    GAN classifier for fraud detection.
    The discriminator is the fraud classifier; the generator acts as an
    adversarial regularizer during training, creating hard examples.

    forward() runs only the discriminator, so it plugs into the same
    evaluate_model() / threshold-sweep code as LSTM and RNN.
    Accepts (batch, n_features) or (batch, 1, n_features) — seq dim squeezed.
    """

    def __init__(self, n_features: int, hidden_size: int = 128,
                 noise_dim: int = 32, dropout: float = 0.3,
                 num_layers: int = 2):
        super().__init__()
        self.noise_dim = noise_dim

        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, n_features),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.discriminator(x).squeeze(-1)


class FraudTransformer(nn.Module):
    """
    Transformer encoder classifier for fraud detection.
    Each feature is treated as one token (seq_len = n_features).
    Accepts (batch, seq_len=1, n_features) like LSTM/RNN — the single
    time-step is expanded to (batch, n_features, 1) so each feature
    becomes a token with embedding size 1, then projected to d_model.
    """

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 hidden_size: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_features) → (batch, n_features, 1)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)          # (batch, n_features, 1)
        else:
            x = x.unsqueeze(-1)             # (batch, n_features, 1)
        x = self.input_proj(x)              # (batch, n_features, d_model)
        x = self.encoder(x)                 # (batch, n_features, d_model)
        x = self.dropout(x.mean(dim=1))     # (batch, d_model) — mean pooling
        return self.fc(x).squeeze(-1)
