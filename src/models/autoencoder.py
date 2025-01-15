import torch as th
from torch import nn
import torch.nn.functional as F
from typing import List
from models import mask_image

class simple_conv(nn.Module):
    def __init__(self, in_channels: int = 3, middle_channels: List[int] = [64, 128, 256], kernel_size: List[int] = [3, 3, 3], stride: List[int] = [2, 2, 2], padding: List[int] = [1, 1, 1], output_padding: List[int] = [1, 1, 1]):
        super(simple_conv, self).__init__()
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

    def forward(self, image: th.tensor, mask: th.tensor):
        masked_img = mask_image(image, mask)
        encoded = self.encoder(masked_img)
        decoded = self.decoder(encoded)
        return decoded

class conv_unet(nn.Module):
    def __init__(self, in_channels: int = 3, middle_channels: List[int] = [64, 128, 256], kernel_size: List[int] = [3, 3, 3], stride: List[int] = [2, 2, 2], padding: List[int] = [1, 1, 1], output_padding: List[int] = [1, 1, 1]):
        super(conv_unet, self).__init__()
        
        self.encoder1 = nn.Conv2d(in_channels, middle_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])  # 64x64 -> 32x32
        self.encoder2 = nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding= padding[1])  # 32x32 -> 16x16
        self.encoder3 = nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])  # 16x16 -> 8x8
        
        self.decoder1 = nn.ConvTranspose2d(middle_channels[2], middle_channels[1], kernel_size=kernel_size[2], stride=stride[2], padding=padding[0], output_padding=output_padding[0])  # 8x8 -> 16x16
        self.decoder2 = nn.ConvTranspose2d(middle_channels[1], middle_channels[0], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1], output_padding=output_padding[1])
        self.decoder3 = nn.ConvTranspose2d(middle_channels[0], in_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[2], output_padding=output_padding[2])
        
        self.relu = nn.ReLU()

    def forward(self, image: th.tensor, mask: th.tensor):
        masked_img = mask_image(image, mask)
        enc1 = self.relu(self.encoder1(masked_img))
        enc2 = self.relu(self.encoder2(enc1))
        enc3 = self.relu(self.encoder3(enc2))
        dec1 = self.relu(self.decoder1(enc3))
        
        dec2 = self.relu(self.decoder2(dec1 + enc2))
        dec3 = nn.Sigmoid()(self.decoder3(dec2 + enc1))
        
        return dec3

class conv_maxpool(nn.Module):
    def __init__(self, in_channels: int, middle_channels: List[int], kernel_size: int = 3, stride: int = 1, pool_size: int = 2, up_kernel: int = 2, up_stride: int = 2, print_sizes: bool=False):
        super(conv_maxpool, self).__init__()
        assert len(middle_channels) == 5, "Middle channels must have 5 elements"
        for c in middle_channels:
            assert isinstance(c, int), "Middle channels must be a list of integers"
            
        self.print_sizes = print_sizes
        
        # Parameters
        activation = nn.ReLU()
        
        # Define encoder
        self.encoder_blocks = nn.ModuleList([
            self._create_conv_block(in_channels, middle_channels[0], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[0], middle_channels[1], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[1], middle_channels[2], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[2], middle_channels[3], kernel_size, stride, activation)
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(pool_size),
            nn.MaxPool2d(pool_size),
            nn.MaxPool2d(pool_size),
            nn.MaxPool2d(pool_size)
        ])
        
        # Define decoder
        self.decoder_blocks = nn.ModuleList([
            self._create_conv_block(middle_channels[3], middle_channels[4], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[4], middle_channels[3], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[3], middle_channels[2], kernel_size, stride, activation),
            self._create_conv_block(middle_channels[2], middle_channels[1], kernel_size, stride, activation)
        ])
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(middle_channels[4], middle_channels[3], up_kernel, up_stride, padding=(up_kernel - 1) // 2),
            nn.ConvTranspose2d(middle_channels[3], middle_channels[2], up_kernel, up_stride, padding=(up_kernel - 1) // 2),
            nn.ConvTranspose2d(middle_channels[2], middle_channels[1], up_kernel, up_stride, padding=(up_kernel - 1) // 2),
            nn.ConvTranspose2d(middle_channels[1], middle_channels[0], up_kernel, up_stride, padding=(up_kernel - 1) // 2)
        ])
        
        # Output layer
        self.output_conv = nn.Conv2d(middle_channels[1], in_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
    
    def _create_conv_block(self, in_channels, out_channels, kernel_size, stride, activation):
        """Helper method to create a convolutional block."""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation,
        )
    
    def forward(self, image: th.tensor, mask: th.tensor) -> th.Tensor:
        
        x = mask_image(image, mask)
        
        encodings = []
        
        # Encoder
        for conv, pool in zip(self.encoder_blocks, self.pools):
            x = conv(x)
            encodings.append(x)
            x = pool(x)
            if self.print_sizes:
                print(f"Encoder block: {x.shape}", flush=True)
        
        # Decoder
        for i, (deconv, upconv) in enumerate(zip(self.decoder_blocks, self.upconvs)):
            x = deconv(x)
            x = upconv(x)
            x = th.cat([x, encodings[-(i + 1)]], dim=1)
            if self.print_sizes:
                print(f"Decoder block {i + 1}: {x.shape}", flush=True)
        
        # Output
        x = self.output_conv(x)
        if self.print_sizes:
            print(f"Output: {x.shape}", flush=True)
        
        return self.sigmoid(x)