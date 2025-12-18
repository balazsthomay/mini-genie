"""
Latent Action Model (LAM)

Learns discrete latent actions from unlabeled video in an unsupervised manner.

Key insight: The encoder sees both frame t and t+1, so it knows what changed.
The decoder only sees frame t + a latent action code, and must reconstruct t+1.
This forces the latent action to encode the meaningful change between frames.

With only 8 possible action codes, the model learns to capture the most
important changes — which empirically end up being movement directions
(left, right, jump, etc.)

Architecture:
- Encoder: CNN that takes (frame_t, frame_t+1) → latent action logits
- VQ layer: discretizes to one of 8 codes
- Decoder: CNN that takes (frame_t, action_embedding) → reconstructed frame_t+1

Note: The decoder is only used for training. At inference, we discard it
and only keep the encoder (for labeling videos) or just the codebook
(for mapping user actions to embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAMEncoder(nn.Module):
    """
    Encodes a pair of frames into a latent action.
    
    Input: frame_t and frame_t+1, each (B, 3, 64, 64)
    Output: latent action logits (B, num_actions)
    
    We concatenate the frames channel-wise and use a CNN to extract
    the action that caused the transition.
    """
    
    def __init__(self, num_actions: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        # Input: concatenated frames (B, 6, 64, 64)
        self.net = nn.Sequential(
            # (B, 6, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(6, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 128, 16, 16) -> (B, 256, 8, 8)
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 256, 8, 8) -> (B, 256, 4, 4)
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GELU(),
            
            # Flatten and project to action logits
            nn.Flatten(),  # (B, 256 * 4 * 4) = (B, 4096)
            nn.Linear(hidden_dim * 4 * 4 * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, num_actions),
        )
    
    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_t: Current frame (B, 3, 64, 64)
            frame_t1: Next frame (B, 3, 64, 64)
        
        Returns:
            action_logits: (B, num_actions)
        """
        # Concatenate frames channel-wise
        x = torch.cat([frame_t, frame_t1], dim=1)  # (B, 6, 64, 64)
        return self.net(x)


class LAMDecoder(nn.Module):
    """
    Reconstructs frame_t+1 from frame_t and an action embedding.
    
    Input: frame_t (B, 3, 64, 64) and action_embedding (B, action_dim)
    Output: reconstructed frame_t+1 (B, 3, 64, 64)
    
    The action embedding is broadcast spatially and concatenated with
    encoded frame features.
    """
    
    def __init__(self, action_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Encode frame_t
        self.encoder = nn.Sequential(
            # (B, 3, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(3, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 128, 16, 16) -> (B, 256, 8, 8)
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GELU(),
        )
        
        # After concatenating action, decode back to image
        # Input channels: hidden_dim * 4 + action_dim (broadcast spatially)
        self.decoder = nn.Sequential(
            # (B, 256 + action_dim, 8, 8) -> (B, 256, 16, 16)
            nn.ConvTranspose2d(hidden_dim * 4 + action_dim, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 256, 16, 16) -> (B, 128, 32, 32)
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 128, 32, 32) -> (B, 64, 64, 64)
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.GELU(),
            
            # (B, 64, 64, 64) -> (B, 3, 64, 64)
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )
    
    def forward(self, frame_t: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_t: Current frame (B, 3, 64, 64)
            action_emb: Action embedding (B, action_dim)
        
        Returns:
            reconstructed: Predicted next frame (B, 3, 64, 64)
        """
        # Encode frame
        h = self.encoder(frame_t)  # (B, 256, 8, 8)
        
        # Broadcast action embedding spatially
        B, C, H, W = h.shape
        action_map = action_emb.view(B, self.action_dim, 1, 1).expand(B, self.action_dim, H, W)
        
        # Concatenate and decode
        h = torch.cat([h, action_map], dim=1)  # (B, 256 + action_dim, 8, 8)
        return self.decoder(h)


class LatentActionModel(nn.Module):
    """
    Complete Latent Action Model.
    
    Training:
        recon, action_indices, vq_loss = lam(frame_t, frame_t1)
        recon_loss = F.mse_loss(recon, frame_t1)
        total_loss = recon_loss + vq_loss
    
    Inference (labeling videos):
        action_indices = lam.encode(frame_t, frame_t1)
    
    Inference (getting action embeddings for dynamics model):
        action_emb = lam.get_action_embedding(action_indices)
    """
    
    def __init__(
        self,
        num_actions: int = 8,
        action_dim: int = 32,
        hidden_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.action_dim = action_dim
        
        self.encoder = LAMEncoder(num_actions=num_actions, hidden_dim=hidden_dim)
        self.decoder = LAMDecoder(action_dim=action_dim, hidden_dim=hidden_dim)
        
        # Action codebook: maps discrete action index to embedding
        self.action_codebook = nn.Embedding(num_actions, action_dim)
        
        self.commitment_cost = commitment_cost
    
    def forward(
        self, 
        frame_t: torch.Tensor, 
        frame_t1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            frame_t: Current frame (B, 3, 64, 64)
            frame_t1: Next frame (B, 3, 64, 64)
        
        Returns:
            recon: Reconstructed next frame (B, 3, 64, 64)
            action_indices: Discrete action indices (B,)
            vq_loss: Vector quantization loss
        """
        # Encode to get action logits
        action_logits = self.encoder(frame_t, frame_t1)  # (B, num_actions)
        
        # Soft action embedding (for gradient flow during training)
        action_probs = F.softmax(action_logits, dim=-1)  # (B, num_actions)
        soft_action_emb = action_probs @ self.action_codebook.weight  # (B, action_dim)
        
        # Hard action selection (discrete)
        action_indices = action_logits.argmax(dim=-1)  # (B,)
        hard_action_emb = self.action_codebook(action_indices)  # (B, action_dim)
        
        # Straight-through estimator: forward uses hard, backward uses soft
        action_emb = soft_action_emb + (hard_action_emb - soft_action_emb).detach()
        
        # Decode to reconstruct next frame
        recon = self.decoder(frame_t, action_emb)
        
        # VQ-style commitment loss: encourage logits to be confident
        # This pushes the soft distribution toward one-hot
        vq_loss = self.commitment_cost * F.mse_loss(soft_action_emb, hard_action_emb.detach())
        
        return recon, action_indices, vq_loss
    
    def encode(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.LongTensor:
        """
        Encode frame pair to discrete action index.
        Used for labeling videos with latent actions.
        
        Args:
            frame_t: Current frame (B, 3, 64, 64)
            frame_t1: Next frame (B, 3, 64, 64)
        
        Returns:
            action_indices: Discrete action indices (B,)
        """
        action_logits = self.encoder(frame_t, frame_t1)
        return action_logits.argmax(dim=-1)
    
    def get_action_embedding(self, action_indices: torch.LongTensor) -> torch.Tensor:
        """
        Get action embeddings from indices.
        Used by dynamics model at training and inference time.
        
        Args:
            action_indices: Action indices (B,) or (B, T)
        
        Returns:
            action_emb: Action embeddings, same shape + (action_dim,)
        """
        return self.action_codebook(action_indices)


# Quick test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    lam = LatentActionModel().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in lam.parameters())
    print(f"LAM parameters: {num_params / 1e6:.2f}M")
    
    # Test forward pass
    frame_t = torch.randn(4, 3, 64, 64).to(device)
    frame_t1 = torch.randn(4, 3, 64, 64).to(device)
    
    recon, action_indices, vq_loss = lam(frame_t, frame_t1)
    
    print(f"Frame t shape: {frame_t.shape}")
    print(f"Frame t+1 shape: {frame_t1.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Action indices: {action_indices}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    
    # Test encode
    actions = lam.encode(frame_t, frame_t1)
    print(f"Encoded actions: {actions}")
    
    # Test get_action_embedding
    emb = lam.get_action_embedding(actions)
    print(f"Action embeddings shape: {emb.shape}")