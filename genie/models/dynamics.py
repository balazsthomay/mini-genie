"""
Dynamics Model for Mini-Genie.

Predicts next frame tokens given previous frame tokens + latent action.
This is the "world model" that simulates what happens when you take an action.

Architecture: Transformer with MaskGIT-style training
- Input: tokens from frame t (16x16 = 256 tokens) + action embedding
- Output: predicted tokens for frame t+1

MaskGIT training:
- Randomly mask some percentage of the target frame's tokens
- Model predicts the masked tokens
- At inference, iteratively unmask tokens over multiple steps

This is simpler than autoregressive generation (predict one token at a time)
and allows parallel prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Standard sinusoidal position embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, seq_len) or (seq_len,) integer positions
        Returns:
            embeddings: (..., dim) position embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicsModel(nn.Module):
    """
    Transformer-based dynamics model with MaskGIT training.
    
    Training:
        loss = dynamics_model(frame_tokens, action_emb, target_tokens)
    
    Inference:
        predicted_tokens = dynamics_model.generate(frame_tokens, action_emb)
    """
    
    def __init__(
        self,
        num_tokens: int = 1024,      # vocabulary size (from video tokenizer)
        num_actions: int = 8,         # number of latent actions
        action_dim: int = 32,         # dimension of action embeddings
        hidden_dim: int = 256,        # transformer hidden dimension
        num_heads: int = 8,
        num_layers: int = 8,
        max_seq_len: int = 256,       # 16x16 tokens per frame
        mask_token_id: int = 1024,    # special mask token (= num_tokens)
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id
        
        # Token embeddings (+1 for mask token)
        self.token_emb = nn.Embedding(num_tokens + 1, hidden_dim)
        
        # Position embeddings (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Action embedding projection
        # We'll add action embedding to all token positions
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output head: predict token logits
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_tokens)
    
    def forward(
        self,
        input_tokens: torch.LongTensor,
        action_emb: torch.Tensor,
        target_tokens: torch.LongTensor,
        mask_ratio: float = None,
    ) -> torch.Tensor:
        """
        Training forward pass with MaskGIT-style masking.
        
        Args:
            input_tokens: Tokens from current frame (B, 256)
            action_emb: Action embedding (B, action_dim)
            target_tokens: Tokens from next frame (B, 256) — what we predict
            mask_ratio: Fraction of target tokens to mask. If None, random [0.5, 1.0]
        
        Returns:
            loss: Cross-entropy loss on masked positions
        """
        B, seq_len = target_tokens.shape
        device = target_tokens.device
        
        # Random mask ratio if not specified
        if mask_ratio is None:
            mask_ratio = torch.empty(1).uniform_(0.5, 1.0).item()
        
        # Create mask: True = masked (need to predict)
        num_masked = int(seq_len * mask_ratio)
        noise = torch.rand(B, seq_len, device=device)
        mask = noise.argsort(dim=1) < num_masked  # (B, seq_len)
        
        # Create input: target tokens with some replaced by mask token
        masked_tokens = target_tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        # Embed tokens
        x = self.token_emb(masked_tokens)  # (B, seq_len, hidden_dim)
        
        # Add position embeddings
        x = x + self.pos_emb[:, :seq_len, :]
        
        # Add action embedding (broadcast to all positions)
        action_feat = self.action_proj(action_emb)  # (B, hidden_dim)
        x = x + action_feat.unsqueeze(1)
        
        # Also add information about the previous frame
        # We concatenate prev frame tokens as conditioning
        prev_emb = self.token_emb(input_tokens)  # (B, seq_len, hidden_dim)
        prev_emb = prev_emb + self.pos_emb[:, :seq_len, :]
        
        # Simple conditioning: add mean of previous frame
        prev_context = prev_emb.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
        x = x + prev_context
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        # Predict logits
        x = self.norm(x)
        logits = self.head(x)  # (B, seq_len, num_tokens)
        
        # Loss only on masked positions
        loss = F.cross_entropy(
            logits[mask].view(-1, self.num_tokens),
            target_tokens[mask].view(-1),
        )
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.LongTensor,
        action_emb: torch.Tensor,
        num_steps: int = 8,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """
        Generate next frame tokens using iterative MaskGIT decoding.
        
        Args:
            input_tokens: Tokens from current frame (B, 256)
            action_emb: Action embedding (B, action_dim)
            num_steps: Number of iterative refinement steps
            temperature: Sampling temperature
        
        Returns:
            predicted_tokens: Generated tokens for next frame (B, 256)
        """
        B, seq_len = input_tokens.shape
        device = input_tokens.device
        
        # Start with all mask tokens
        tokens = torch.full((B, seq_len), self.mask_token_id, device=device)
        
        # Precompute conditioning from previous frame
        prev_emb = self.token_emb(input_tokens)
        prev_emb = prev_emb + self.pos_emb[:, :seq_len, :]
        prev_context = prev_emb.mean(dim=1, keepdim=True)
        action_feat = self.action_proj(action_emb).unsqueeze(1)
        
        # Track which positions are still masked
        is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        
        for step in range(num_steps):
            # Embed current tokens
            x = self.token_emb(tokens)
            x = x + self.pos_emb[:, :seq_len, :]
            x = x + action_feat
            x = x + prev_context
            
            # Transformer
            for block in self.blocks:
                x = block(x)
            
            # Predict logits
            x = self.norm(x)
            logits = self.head(x)  # (B, seq_len, num_tokens)
            
            # Sample from logits
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Get confidence scores (max prob) for masked positions
            confidence = probs.max(dim=-1).values  # (B, seq_len)
            confidence[~is_masked] = float('inf')  # don't re-predict unmasked
            
            # Determine how many to unmask this step
            # Linear schedule: unmask more as we go
            num_to_unmask = int(seq_len * (step + 1) / num_steps) - int(seq_len * step / num_steps)
            num_to_unmask = max(1, num_to_unmask)
            
            # For each batch, find top-k most confident masked positions
            for b in range(B):
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue
                
                # Get confidences for masked positions
                masked_conf = confidence[b, masked_indices]
                
                # Select top-k most confident
                k = min(num_to_unmask, len(masked_indices))
                _, top_k_idx = masked_conf.topk(k)
                unmask_positions = masked_indices[top_k_idx]
                
                # Sample tokens for these positions
                for pos in unmask_positions:
                    sampled = torch.multinomial(probs[b, pos], 1).item()
                    tokens[b, pos] = sampled
                    is_masked[b, pos] = False
        
        return tokens


# Quick test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    model = DynamicsModel().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Dynamics model parameters: {num_params / 1e6:.2f}M")
    
    # Test forward pass (training)
    B = 4
    input_tokens = torch.randint(0, 1024, (B, 256)).to(device)
    action_emb = torch.randn(B, 32).to(device)
    target_tokens = torch.randint(0, 1024, (B, 256)).to(device)
    
    loss = model(input_tokens, action_emb, target_tokens)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test generate (inference)
    generated = model.generate(input_tokens, action_emb, num_steps=8)
    print(f"Generated tokens shape: {generated.shape}")
    print(f"Generated token range: [{generated.min()}, {generated.max()}]")