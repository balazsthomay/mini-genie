"""
VQ-VAE Video Tokenizer

Compresses video frames into discrete tokens using vector quantization.
Simpler CNN-based architecture (paper uses ST-Transformer, but CNN is faster).

Key concepts:
- Encoder: image (3, 64, 64) -> continuous latent (32, 16, 16)
- Quantizer: snap each spatial position to nearest codebook vector
- Decoder: discrete tokens -> reconstructed image

The discrete tokens are what the dynamics model will predict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    """Simple residual block with two convolutions."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class Encoder(nn.Module):
    """
    Encodes images to continuous latent representations.
    
    Input: (B, 3, 64, 64)
    Output: (B, latent_dim, 16, 16)
    
    Spatial compression is 4x. Channels increase as spatial dims decrease.
    """
    
    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        # Channel progression: hidden_dim -> hidden_dim*2 -> hidden_dim*4
        dim1 = hidden_dim      # 64
        dim2 = hidden_dim * 2  # 128
        dim3 = hidden_dim * 4  # 256
        
        self.net = nn.Sequential(
            # Initial projection: (B, 3, 64, 64) -> (B, dim1, 64, 64)
            nn.Conv2d(in_channels, dim1, 3, padding=1),
            nn.GELU(),
            ResBlock(dim1),
            
            # Downsample 1: (B, dim1, 64, 64) -> (B, dim2, 32, 32)
            nn.Conv2d(dim1, dim2, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(dim2),
            
            # Downsample 2: (B, dim2, 32, 32) -> (B, dim3, 16, 16)
            nn.Conv2d(dim2, dim3, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(dim3),
            
            # Project to latent dim: (B, dim3, 16, 16) -> (B, latent_dim, 16, 16)
            nn.Conv2d(dim3, latent_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Decodes quantized latents back to images.
    
    Input: (B, latent_dim, 16, 16)
    Output: (B, 3, 64, 64)
    
    Mirrors encoder: channels decrease as spatial dims increase.
    """
    
    def __init__(self, out_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        # Channel progression (reverse of encoder): dim3 -> dim2 -> dim1
        dim1 = hidden_dim      # 64
        dim2 = hidden_dim * 2  # 128
        dim3 = hidden_dim * 4  # 256
        
        self.net = nn.Sequential(
            # Project from latent dim: (B, latent_dim, 16, 16) -> (B, dim3, 16, 16)
            nn.Conv2d(latent_dim, dim3, 1),
            nn.GELU(),
            ResBlock(dim3),
            
            # Upsample 1: (B, dim3, 16, 16) -> (B, dim2, 32, 32)
            nn.ConvTranspose2d(dim3, dim2, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(dim2),
            
            # Upsample 2: (B, dim2, 32, 32) -> (B, dim1, 64, 64)
            nn.ConvTranspose2d(dim2, dim1, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(dim1),
            
            # Final projection: (B, dim1, 64, 64) -> (B, 3, 64, 64)
            nn.Conv2d(dim1, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.
    
    Maintains a codebook of learned vectors. Each spatial position in the
    encoder output is replaced with the nearest codebook vector.
    
    This creates the discrete bottleneck that allows us to treat video
    prediction as a token prediction problem.
    """
    
    def __init__(self, num_codes: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: (num_codes, latent_dim)
        # Each row is a learnable vector that tokens can snap to
        self.codebook = nn.Embedding(num_codes, latent_dim)
        
        # Initialize codebook with small random values
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Args:
            z: Encoder output, shape (B, latent_dim, H, W)
        
        Returns:
            z_q: Quantized latents, shape (B, latent_dim, H, W)
            loss: Commitment loss (for training)
            indices: Codebook indices, shape (B, H, W)
        """
        # Rearrange to (B, H, W, latent_dim) for easier distance computation
        z = rearrange(z, 'b c h w -> b h w c')
        z_flat = rearrange(z, 'b h w c -> (b h w) c')
        
        # Compute distances to all codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        codebook = self.codebook.weight  # (num_codes, latent_dim)
        
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)           # (BHW, 1)
            + codebook.pow(2).sum(dim=1, keepdim=True).T     # (1, num_codes)
            - 2 * z_flat @ codebook.T                        # (BHW, num_codes)
        )
        
        # Find nearest codebook entry for each position
        indices = d.argmin(dim=1)  # (BHW,)
        
        # Get quantized vectors
        z_q_flat = self.codebook(indices)  # (BHW, latent_dim)
        
        # Reshape back
        b, h, w, c = z.shape
        indices = indices.view(b, h, w)
        z_q = z_q_flat.view(b, h, w, c)
        
        # Compute losses
        # Codebook loss: move codebook vectors toward encoder outputs
        codebook_loss = F.mse_loss(z_q, z.detach())
        # Commitment loss: encourage encoder outputs to stay close to codebook
        commitment_loss = F.mse_loss(z, z_q.detach())
        
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        # This is the trick that lets us backprop through the discrete bottleneck
        z_q = z + (z_q - z).detach()
        
        # Rearrange back to (B, latent_dim, H, W)
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        
        return z_q, loss, indices


class VideoTokenizer(nn.Module):
    """
    Complete VQ-VAE tokenizer.
    
    Usage:
        tokenizer = VideoTokenizer(...)
        
        # Training: get reconstruction and loss
        recon, vq_loss, indices = tokenizer(images)
        recon_loss = F.mse_loss(recon, images)
        total_loss = recon_loss + vq_loss
        
        # Encoding: images -> tokens
        tokens = tokenizer.encode(images)  # (B, H, W) integers
        
        # Decoding: tokens -> images  
        images = tokenizer.decode(tokens)  # (B, 3, 64, 64)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_codes: int = 1024,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim)
        
        self.num_codes = num_codes
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Full forward pass for training.
        
        Args:
            x: Images, shape (B, 3, H, W), values in [0, 1]
        
        Returns:
            recon: Reconstructed images, shape (B, 3, H, W)
            vq_loss: Vector quantization loss
            indices: Token indices, shape (B, h, w)
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices
    
    def encode(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Encode images to discrete tokens.
        
        Args:
            x: Images, shape (B, 3, H, W)
        
        Returns:
            indices: Token indices, shape (B, h, w)
        """
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices
    
    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        Decode tokens back to images.
        
        Args:
            indices: Token indices, shape (B, h, w)
        
        Returns:
            images: Reconstructed images, shape (B, 3, H, W)
        """
        # Look up codebook vectors
        z_q = self.quantizer.codebook(indices)  # (B, h, w, latent_dim)
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return self.decoder(z_q)


# Quick test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    tokenizer = VideoTokenizer().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"Tokenizer parameters: {num_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64).to(device)
    recon, vq_loss, indices = tokenizer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Token indices shape: {indices.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Token vocab used: {indices.unique().shape[0]} / {tokenizer.num_codes}")
    
    # Test encode/decode
    tokens = tokenizer.encode(x)
    decoded = tokenizer.decode(tokens)
    print(f"Encode->Decode shape: {decoded.shape}")