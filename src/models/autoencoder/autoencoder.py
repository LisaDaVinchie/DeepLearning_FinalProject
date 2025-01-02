from torch import nn
import torch.nn.functional as F

class Autoencoder_conv(nn.Module):
    def __init__(self, in_channels=3, middle_channels= [64, 128, 256], kernel_size = [3, 3, 3], stride = [2, 2, 2], padding = [1, 1, 1], output_padding = [1, 1, 1]):
        super(Autoencoder_conv, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding= padding[1]),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2]),  # 16x16 -> 8x8
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(middle_channels[2], middle_channels[1], kernel_size=kernel_size[2], stride=stride[2], padding=padding[0], output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels[1], middle_channels[0], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1], output_padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels[0], in_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[2], output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Autoencoder_unet(nn.Module):
    def __init__(self, in_channels=3, middle_channels= [64, 128, 256], kernel_size = [3, 3, 3], stride = [2, 2, 2], padding = [1, 1, 1], output_padding = [1, 1, 1]):
        super(Autoencoder_unet, self).__init__()
        
        self.encoder1 = nn.Conv2d(in_channels, middle_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])  # 64x64 -> 32x32
        self.encoder2 = nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding= padding[1])  # 32x32 -> 16x16
        self.encoder3 = nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])  # 16x16 -> 8x8
        
        self.decoder1 = nn.ConvTranspose2d(middle_channels[2], middle_channels[1], kernel_size=kernel_size[2], stride=stride[2], padding=padding[0], output_padding=1)  # 8x8 -> 16x16
        self.decoder2 = nn.ConvTranspose2d(middle_channels[1], middle_channels[0], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1], output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(middle_channels[0], in_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[2], output_padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        enc1 = self.relu(self.encoder1(x))
        enc2 = self.relu(self.encoder2(enc1))
        enc3 = self.relu(self.encoder3(enc2))
        dec1 = self.relu(self.decoder1(enc3))
        
        dec2 = self.relu(self.decoder2(dec1 + enc2))
        dec3 = nn.Sigmoid()(self.decoder3(dec2 + enc1))
        
        return dec3
