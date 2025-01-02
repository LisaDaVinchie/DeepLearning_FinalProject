import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: th.Tensor):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class TransformerInpainting(nn.Module):
    def __init__(self, img_size=64, patch_size=16, embed_dim=1024, num_heads=16, num_layers=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = 3
        
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(th.zeros(1, patch_size, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, patch_size * patch_size * self.n_channels)  # Output patch pixel values
        
    def reconstruct_image(self, decoded_patches, img_size, patch_size):
        B, n_patches, patch_dim = decoded_patches.shape
        C = patch_dim // (patch_size * patch_size)
        H, W = img_size, img_size  # Assuming square images

        # Reshape decoded patches
        patches = decoded_patches.view(B, n_patches, C, patch_size, patch_size)

        # Rearrange patches into a grid
        n_row_patches = H // patch_size
        n_col_patches = W // patch_size
        image = patches.view(B, n_row_patches, n_col_patches, C, patch_size, patch_size)
        image = image.permute(0, 3, 1, 4, 2, 5)  # Reorder dimensions
        image = image.reshape(B, C, H, W)  # Merge patches into the final image
        return image


    def forward(self, x: th.Tensor, mask: th.Tensor):
        patches = self.patch_embed(x)  # [B, n_patches, embed_dim]
        patches += self.pos_embed
        mask = mask.unsqueeze(1).repeat(1, self.n_channels, 1, 1)
        # Downsample the mask to match the patch grid size
        mask = F.interpolate(mask.float(), scale_factor=1/self.patch_size, mode='nearest')  # Shape: [B, C, H/patch_size, W/patch_size]
        mask = mask.flatten(2).transpose(1, 2)  # [B, n_patches, C]
        masked_patches = patches * (~mask.bool().all(dim=-1, keepdim=True))  # Zero-out masked patches
        encoded = self.encoder(masked_patches)
        decoded_patches = self.fc(encoded)  # [B, n_patches, patch_dim]
        return self.reconstruct_image(decoded_patches, self.img_size, self.patch_size)