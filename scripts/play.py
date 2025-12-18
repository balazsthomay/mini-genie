#!/usr/bin/env python
"""
Play script for Mini-Genie.

Generates interactive environments from a starting frame.
User provides actions (0-7), model generates next frames.

Usage:
    python scripts/play.py --output_dir outputs --prompt_image path/to/image.png
    
Or use a random frame from the dataset:
    python scripts/play.py --output_dir outputs --use_dataset_frame
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.models import VideoTokenizer, LatentActionModel, DynamicsModel
from configs.default import get_config


def load_models(output_dir: str, device: str):
    """Load all trained models."""
    config = get_config()
    
    # Load tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer_final.pt")
    print(f"Loading tokenizer from {tokenizer_path}")
    checkpoint = torch.load(tokenizer_path, map_location=device)
    
    tokenizer = VideoTokenizer(
        in_channels=config.tokenizer.in_channels,
        hidden_dim=config.tokenizer.hidden_dim,
        latent_dim=config.tokenizer.latent_dim,
        num_codes=config.tokenizer.num_codes,
    ).to(device)
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    tokenizer.eval()
    
    # Load LAM and Dynamics
    lam_dyn_path = os.path.join(output_dir, "lam_dynamics_final.pt")
    print(f"Loading LAM + Dynamics from {lam_dyn_path}")
    checkpoint = torch.load(lam_dyn_path, map_location=device)
    
    lam = LatentActionModel(
        num_actions=config.lam.num_actions,
        action_dim=config.lam.action_dim,
        hidden_dim=config.lam.hidden_dim,
    ).to(device)
    lam.load_state_dict(checkpoint['lam_state_dict'])
    lam.eval()
    
    dynamics = DynamicsModel(
        num_tokens=config.tokenizer.num_codes,
        num_actions=config.lam.num_actions,
        action_dim=config.lam.action_dim,
        hidden_dim=config.dynamics.hidden_dim,
        num_heads=config.dynamics.num_heads,
        num_layers=config.dynamics.num_layers,
    ).to(device)
    dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
    dynamics.eval()
    
    return tokenizer, lam, dynamics


def load_prompt_image(image_path: str, size: int = 64) -> torch.Tensor:
    """Load and preprocess a prompt image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    return img.unsqueeze(0)  # (1, 3, H, W)


def load_dataset_frame(data_dir: str, frame_idx: int = None) -> torch.Tensor:
    """Load a frame from the dataset."""
    frames_path = os.path.join(data_dir, "frames.npy")
    frames = np.load(frames_path)  # (N, H, W, 3)
    
    if frame_idx is None:
        frame_idx = np.random.randint(0, len(frames))
    
    print(f"Using frame {frame_idx} from dataset")
    frame = frames[frame_idx].astype(np.float32) / 255.0
    frame = torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
    return frame.unsqueeze(0)  # (1, 3, H, W)


@torch.no_grad()
def generate_step(
    current_frame: torch.Tensor,
    action: int,
    tokenizer: VideoTokenizer,
    lam: LatentActionModel,
    dynamics: DynamicsModel,
    device: str,
) -> torch.Tensor:
    """
    Generate the next frame given current frame and action.
    
    Args:
        current_frame: (1, 3, 64, 64) tensor
        action: Integer action 0-7
        tokenizer: Trained video tokenizer
        lam: Trained LAM (only need codebook)
        dynamics: Trained dynamics model
        device: Device
    
    Returns:
        next_frame: (1, 3, 64, 64) tensor
    """
    current_frame = current_frame.to(device)
    
    # Tokenize current frame
    tokens = tokenizer.encode(current_frame)  # (1, 16, 16)
    tokens_flat = tokens.view(1, -1)  # (1, 256)
    
    # Get action embedding
    action_tensor = torch.tensor([action], device=device)
    action_emb = lam.get_action_embedding(action_tensor)  # (1, action_dim)
    
    # Generate next frame tokens
    next_tokens = dynamics.generate(
        tokens_flat, 
        action_emb,
        num_steps=8,
        temperature=1.0,
    )  # (1, 256)
    
    # Reshape and decode
    next_tokens = next_tokens.view(1, 16, 16)
    next_frame = tokenizer.decode(next_tokens)  # (1, 3, 64, 64)
    
    return next_frame


def save_frame(frame: torch.Tensor, path: str):
    """Save a frame tensor as an image."""
    frame = frame.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    frame = (frame.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    Image.fromarray(frame).save(path)


def interactive_play(
    tokenizer: VideoTokenizer,
    lam: LatentActionModel,
    dynamics: DynamicsModel,
    initial_frame: torch.Tensor,
    device: str,
    output_dir: str,
):
    """Interactive play loop."""
    print("\n" + "=" * 50)
    print("Mini-Genie Interactive Play")
    print("=" * 50)
    print("\nActions:")
    print("  0-7: Take action (meaning learned from data)")
    print("  q: Quit")
    print("  r: Reset to initial frame")
    print("  s: Save current frame")
    print("\n")
    
    current_frame = initial_frame.clone()
    frame_count = 0
    
    # Save initial frame
    os.makedirs(output_dir, exist_ok=True)
    save_frame(current_frame, os.path.join(output_dir, "frame_000.png"))
    print(f"Saved initial frame to {output_dir}/frame_000.png")
    
    while True:
        user_input = input("Enter action (0-7, q=quit, r=reset, s=save): ").strip().lower()
        
        if user_input == 'q':
            print("Goodbye!")
            break
        elif user_input == 'r':
            current_frame = initial_frame.clone()
            frame_count = 0
            print("Reset to initial frame")
            continue
        elif user_input == 's':
            save_path = os.path.join(output_dir, f"saved_{frame_count:03d}.png")
            save_frame(current_frame, save_path)
            print(f"Saved to {save_path}")
            continue
        
        try:
            action = int(user_input)
            if not 0 <= action <= 7:
                print("Action must be 0-7")
                continue
        except ValueError:
            print("Invalid input")
            continue
        
        # Generate next frame
        print(f"Generating with action {action}...")
        current_frame = generate_step(
            current_frame, action, tokenizer, lam, dynamics, device
        )
        frame_count += 1
        
        # Save frame
        save_path = os.path.join(output_dir, f"frame_{frame_count:03d}.png")
        save_frame(current_frame, save_path)
        print(f"Saved frame {frame_count} to {save_path}")


def batch_generate(
    tokenizer: VideoTokenizer,
    lam: LatentActionModel,
    dynamics: DynamicsModel,
    initial_frame: torch.Tensor,
    actions: list,
    device: str,
    output_dir: str,
):
    """Generate a sequence of frames from a list of actions."""
    os.makedirs(output_dir, exist_ok=True)
    
    current_frame = initial_frame.clone()
    frames = [current_frame.clone()]
    
    save_frame(current_frame, os.path.join(output_dir, "frame_000.png"))
    
    for i, action in enumerate(actions):
        print(f"Step {i+1}/{len(actions)}: action={action}")
        current_frame = generate_step(
            current_frame, action, tokenizer, lam, dynamics, device
        )
        frames.append(current_frame.clone())
        save_frame(current_frame, os.path.join(output_dir, f"frame_{i+1:03d}.png"))
    
    print(f"\nGenerated {len(actions)} frames, saved to {output_dir}/")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Play Mini-Genie")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory with trained model checkpoints")
    parser.add_argument("--prompt_image", type=str, default=None,
                        help="Path to prompt image")
    parser.add_argument("--data_dir", type=str, default="data/coinrun",
                        help="Dataset directory (for loading dataset frames)")
    parser.add_argument("--use_dataset_frame", action="store_true",
                        help="Use a random frame from dataset as prompt")
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="Specific frame index from dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated actions for batch generation (e.g., '2,2,2,5,5')")
    parser.add_argument("--play_output", type=str, default="play_output",
                        help="Directory to save generated frames")
    
    args = parser.parse_args()
    
    # Auto-detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    tokenizer, lam, dynamics = load_models(args.output_dir, device)
    
    # Load initial frame
    if args.prompt_image:
        initial_frame = load_prompt_image(args.prompt_image)
    elif args.use_dataset_frame or args.frame_idx is not None:
        initial_frame = load_dataset_frame(args.data_dir, args.frame_idx)
    else:
        print("No prompt specified, using random dataset frame")
        initial_frame = load_dataset_frame(args.data_dir)
    
    # Run
    if args.actions:
        # Batch generation
        actions = [int(a.strip()) for a in args.actions.split(',')]
        batch_generate(
            tokenizer, lam, dynamics, initial_frame, actions, device, args.play_output
        )
    else:
        # Interactive mode
        interactive_play(
            tokenizer, lam, dynamics, initial_frame, device, args.play_output
        )


if __name__ == "__main__":
    main()