"""
Training script for LAM + Dynamics Model (jointly).

This is Phase 2 of training — requires trained tokenizer from Phase 1.

The LAM learns to:
- Extract latent actions from frame pairs (unsupervised)

The Dynamics model learns to:
- Predict next frame tokens given current tokens + latent action

Both are trained jointly:
1. LAM encoder sees (frame_t, frame_t+1) → latent action
2. LAM decoder reconstructs frame_t+1 from (frame_t, action) → LAM loss
3. Dynamics predicts tokens_t+1 from (tokens_t, action) → Dynamics loss
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genie.models import VideoTokenizer, LatentActionModel, DynamicsModel
from genie.data import CoinRunDataset
from configs.default import get_config


def train_dynamics(
    data_dir: str = None,
    output_dir: str = None,
    tokenizer_path: str = None,
    num_steps: int = None,
    batch_size: int = None,
    lr: float = None,
    device: str = None,
    log_every: int = None,
    save_every: int = None,
):
    """
    Train LAM and Dynamics model jointly.
    
    Args:
        data_dir: Directory containing frames.npy
        output_dir: Directory to save checkpoints
        tokenizer_path: Path to trained tokenizer checkpoint
        num_steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        log_every: Log every N steps
        save_every: Save checkpoint every N steps
    """
    # Load config and override with any provided args
    config = get_config()
    
    data_dir = data_dir or config.data.data_dir
    output_dir = output_dir or config.output_dir
    tokenizer_path = tokenizer_path or os.path.join(output_dir, "tokenizer_final.pt")
    num_steps = num_steps or config.dynamics.num_steps
    batch_size = batch_size or config.dynamics.batch_size
    lr = lr or config.dynamics.lr
    device = device or config.device
    log_every = log_every or config.dynamics.log_every
    save_every = save_every or config.dynamics.save_every
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer (frozen)
    print(f"Loading tokenizer from {tokenizer_path}")
    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=False)
    tokenizer = VideoTokenizer(
        in_channels=config.tokenizer.in_channels,
        hidden_dim=config.tokenizer.hidden_dim,
        latent_dim=config.tokenizer.latent_dim,
        num_codes=config.tokenizer.num_codes,
    ).to(device)
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    tokenizer.eval()  # Frozen
    for param in tokenizer.parameters():
        param.requires_grad = False
    print("Tokenizer loaded and frozen")
    
    # Load dataset (sequences of frames)
    print(f"Loading data from {data_dir}")
    dataset = CoinRunDataset(
        data_dir=data_dir, 
        seq_len=config.data.seq_len,
        split="train"
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    
    # Create models
    lam = LatentActionModel(
        num_actions=config.lam.num_actions,
        action_dim=config.lam.action_dim,
        hidden_dim=config.lam.hidden_dim,
    ).to(device)
    
    dynamics = DynamicsModel(
        num_tokens=config.tokenizer.num_codes,
        num_actions=config.lam.num_actions,
        action_dim=config.lam.action_dim,
        hidden_dim=config.dynamics.hidden_dim,
        num_heads=config.dynamics.num_heads,
        num_layers=config.dynamics.num_layers,
    ).to(device)
    
    lam_params = sum(p.numel() for p in lam.parameters())
    dyn_params = sum(p.numel() for p in dynamics.parameters())
    print(f"LAM parameters: {lam_params / 1e6:.2f}M")
    print(f"Dynamics parameters: {dyn_params / 1e6:.2f}M")
    
    # Optimizer (joint)
    optimizer = torch.optim.AdamW(
        list(lam.parameters()) + list(dynamics.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    
    # Training loop
    lam.train()
    dynamics.train()
    pbar = tqdm(range(num_steps), desc="Training LAM + Dynamics")
    
    running_lam_loss = 0.0
    running_dyn_loss = 0.0
    
    for step in pbar:
        # Get batch (infinite iterator)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # batch shape: (B, seq_len, 3, 64, 64)
        frames = batch.to(device)
        B, T, C, H, W = frames.shape
        
        # We'll use consecutive frame pairs for training
        # frame_t -> frame_t+1 for each t in [0, T-2]
        
        total_lam_loss = 0.0
        total_dyn_loss = 0.0
        num_pairs = T - 1
        
        for t in range(num_pairs):
            frame_t = frames[:, t]      # (B, 3, 64, 64)
            frame_t1 = frames[:, t + 1]  # (B, 3, 64, 64)
            
            # === LAM forward ===
            recon, action_indices, vq_loss = lam(frame_t, frame_t1)
            lam_recon_loss = F.mse_loss(recon, frame_t1)
            lam_loss = lam_recon_loss + vq_loss
            total_lam_loss += lam_loss
            
            # === Dynamics forward ===
            # Tokenize frames (frozen tokenizer)
            with torch.no_grad():
                tokens_t = tokenizer.encode(frame_t)    # (B, 16, 16)
                tokens_t1 = tokenizer.encode(frame_t1)  # (B, 16, 16)
            
            # Flatten spatial dims
            tokens_t_flat = tokens_t.view(B, -1)    # (B, 256)
            tokens_t1_flat = tokens_t1.view(B, -1)  # (B, 256)
            
            # Get action embedding (detached from LAM for dynamics)
            action_emb = lam.get_action_embedding(action_indices)  # (B, action_dim)
            
            # Dynamics predicts next frame tokens
            dyn_loss = dynamics(tokens_t_flat, action_emb, tokens_t1_flat)
            total_dyn_loss += dyn_loss
        
        # Average over frame pairs
        total_lam_loss = total_lam_loss / num_pairs
        total_dyn_loss = total_dyn_loss / num_pairs
        
        # Combined loss
        loss = total_lam_loss + total_dyn_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        running_lam_loss += total_lam_loss.item()
        running_dyn_loss += total_dyn_loss.item()
        
        # Logging
        if (step + 1) % log_every == 0:
            avg_lam = running_lam_loss / log_every
            avg_dyn = running_dyn_loss / log_every
            
            pbar.set_postfix({
                'lam': f'{avg_lam:.4f}',
                'dyn': f'{avg_dyn:.4f}',
            })
            
            running_lam_loss = 0.0
            running_dyn_loss = 0.0
        
        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"lam_dynamics_step{step+1}.pt")
            torch.save({
                'step': step + 1,
                'lam_state_dict': lam.state_dict(),
                'dynamics_state_dict': dynamics.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Save final models
    final_path = os.path.join(output_dir, "lam_dynamics_final.pt")
    torch.save({
        'step': num_steps,
        'lam_state_dict': lam.state_dict(),
        'dynamics_state_dict': dynamics.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"Saved final models to {final_path}")
    
    return lam, dynamics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LAM + Dynamics")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    train_dynamics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )