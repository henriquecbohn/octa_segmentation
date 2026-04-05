import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # Primeira convolução
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),   # estabiliza o treino
            nn.ReLU(inplace=True),
            # Segunda convolução
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # reduz resolução pela metade

    def forward(self, x):
        skip = self.conv(x)   # guarda para skip connection
        x    = self.pool(skip) # reduz resolução
        return skip, x


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)  # aumenta resolução


        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip, x], dim=1)  # concatena no eixo dos canais
        x = self.conv(x)
        return x


class UNet(nn.Module):
  
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
 
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return torch.sigmoid(self.final_conv(x))

# TESTE RÁPIDO

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")

    model = UNet(in_channels=1, out_channels=1).to(device)

    x = torch.randn(4, 1, 400, 400).to(device)
    y = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Min/Max saída: {y.min():.3f} / {y.max():.3f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis: {total_params:,}")
    print("✅ U-Net funcionando!")