"""
Training script for Video Tokenizer (VQ-VAE).

This is Phase 1 of training — must complete before training dynamics model.

The tokenizer learns to:
1. Compress 64x64 images into 16x16 discrete tokens
2. Reconstruct images from those tokens

Loss = reconstruction_loss + vq_commitment_loss
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genie.models import VideoTokenizer
from genie.data import SingleFrameDataset
from configs.default import get_config


def train_tokenizer(
    data_dir: str = None,
    output_dir: str = None,
    num_steps: int = None,
    batch_size: int = None,
    lr: float = None,
    device: str = None,
    log_every: int = None,
    save_every: int = None,
):
    """
    Train the video tokenizer.
    
    Args:
        data_dir: Directory containing frames.npy
        output_dir: Directory to save checkpoints
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
    num_steps = num_steps or config.tokenizer.num_steps
    batch_size = batch_size or config.tokenizer.batch_size
    lr = lr or config.tokenizer.lr
    device = device or config.device
    log_every = log_every or config.tokenizer.log_every
    save_every = save_every or config.tokenizer.save_every
    
    # Auto-detect device
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading data from {data_dir}")
    dataset = SingleFrameDataset(data_dir=data_dir, split="train")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    
    # Create model
    model = VideoTokenizer(
        in_channels=config.tokenizer.in_channels,
        hidden_dim=config.tokenizer.hidden_dim,
        latent_dim=config.tokenizer.latent_dim,
        num_codes=config.tokenizer.num_codes,
        commitment_cost=config.tokenizer.commitment_cost,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"VideoTokenizer parameters: {num_params / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    model.train()
    pbar = tqdm(range(num_steps), desc="Training Tokenizer")
    
    running_recon_loss = 0.0
    running_vq_loss = 0.0
    
    for step in pbar:
        # Get batch (infinite iterator)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        frames = batch.to(device)  # (B, 3, 64, 64)
        
        # Forward pass
        recon, vq_loss, indices = model(frames)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, frames)
        
        # Total loss
        loss = recon_loss + vq_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        running_recon_loss += recon_loss.item()
        running_vq_loss += vq_loss.item()
        
        # Logging
        if (step + 1) % log_every == 0:
            avg_recon = running_recon_loss / log_every
            avg_vq = running_vq_loss / log_every
            
            # Count unique codes used in this batch
            unique_codes = indices.unique().shape[0]
            
            pbar.set_postfix({
                'recon': f'{avg_recon:.4f}',
                'vq': f'{avg_vq:.4f}',
                'codes': f'{unique_codes}/{model.num_codes}',
            })
            
            running_recon_loss = 0.0
            running_vq_loss = 0.0
        
        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"tokenizer_step{step+1}.pt")
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "tokenizer_final.pt")
    torch.save({
        'step': num_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Video Tokenizer")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    train_tokenizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )