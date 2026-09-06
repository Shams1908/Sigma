import torch
import torch.nn as nn

class RawIQCNN(nn.Module):
    """
    reproducible baseline 1D CNN for modulation classification from raw IQ waveforms.
    Input shape: [batch, 2, 128]
    Output shape: [batch, 11]
    """
    def __init__(self, num_classes: int = 11):
        super(RawIQCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 128 -> 64
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 64 -> 32
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Fully Connected Head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2, 128].
            
        Returns:
            torch.Tensor: Output logits of shape [batch_size, 11].
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global Average Pooling (GAP) across the time dimension
        x = x.mean(dim=-1) # [batch, 256, 32] -> [batch, 256]
        
        x = self.fc(x)
        return x
