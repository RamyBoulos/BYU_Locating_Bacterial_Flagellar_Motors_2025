import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A block with two consecutive 3D convolutional layers followed by ReLU activations.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for DoubleConv.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, D, H, W)
        """
        return self.conv(x)


class UNet3D(nn.Module):
    """
    A standard 3D U-Net architecture for volumetric image-to-image tasks.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for grayscale).
    out_channels : int
        Number of output channels (e.g., 1 for regression heatmap).
    features : list of int
        List of feature map sizes for each level.
    final_activation : str
        Output activation to apply ('identity', 'sigmoid', 'softmax', or 'tanh').
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], final_activation="identity"):
        super(UNet3D, self).__init__()
        self.final_activation = final_activation
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down path
        in_chs = in_channels
        for feat in features:
            self.downs.append(DoubleConv(in_chs, feat))
            in_chs = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up path
        up_feats = list(reversed(features))
        for feat in up_feats:
            self.ups.append(
                nn.ConvTranspose3d(feat*2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat*2, feat))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for 3D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, D, H, W)
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx // 2]
            # Handle shape mismatch due to odd input sizes
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx+1](x)
        out = self.final_conv(x)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.final_activation == "softmax":
            return torch.softmax(out, dim=1)
        elif self.final_activation == "tanh":
            return torch.tanh(out)
        return out