import torch
import torch.nn as nn


class RawIQCNN(nn.Module):
    """
    1D CNN for modulation classification from raw IQ waveforms.

    Input shape: [batch, in_channels, 128]
    Output shape: [batch, num_classes]

    Default (backward-compatible):
        in_channels=2  → RAW_IQ [I, Q]   (M5 baseline model)

    M6 multi-channel:
        in_channels=3  → IQ_AMPLITUDE [I, Q, |z|]
        in_channels=4  → IQ_AMP_PHASE [I, Q, |z|, Δφ]
        in_channels=2  → AMPLITUDE_PHASE [|z|, Δφ]  (same 2-ch, different meaning)

    Checkpoint compatibility:
        2-channel checkpoints (M5) can be loaded directly.
        3- or 4-channel checkpoints carry ``in_channels`` in their metadata
        so the model is reconstructed with the correct first-layer width.
    """

    def __init__(self, num_classes: int = 11, in_channels: int = 2) -> None:
        super(RawIQCNN, self).__init__()

        self.in_channels = in_channels

        # Block 1 — first conv uses configurable in_channels
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 -> 32
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Fully Connected Head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape [batch_size, in_channels, 128].

        Returns:
            Logits of shape [batch_size, num_classes].
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Global Average Pooling across the time dimension
        x = x.mean(dim=-1)  # [batch, 256, 32] -> [batch, 256]
        x = self.fc(x)
        return x
