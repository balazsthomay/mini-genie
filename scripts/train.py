#!/usr/bin/env python
"""
Main training script for Mini-Genie.

Runs the full training pipeline:
1. Collect data (if needed)
2. Train video tokenizer
3. Train LAM + dynamics model

Usage:
    python scripts/train.py --data_dir data/coinrun --output_dir outputs
    
Or run phases separately:
    python scripts/train.py --phase tokenizer
    python scripts/train.py --phase dynamics --tokenizer_path outputs/tokenizer_final.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default import get_config


def main():
    parser = argparse.ArgumentParser(description="Train Mini-Genie")
    parser.add_argument("--phase", type=str, default="all", 
                        choices=["all", "collect", "tokenizer", "dynamics"],
                        help="Which phase to run")
    parser.add_argument("--data_dir", type=str, default="data/coinrun")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to trained tokenizer (for dynamics phase)")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Override config values
    parser.add_argument("--tokenizer_steps", type=int, default=None)
    parser.add_argument("--dynamics_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    
    args = parser.parse_args()
    
    config = get_config()
    
    # Phase 1: Collect data
    if args.phase in ["all", "collect"]:
        print("=" * 50)
        print("Phase 0: Collecting CoinRun data")
        print("=" * 50)
        
        from genie.data import collect_coinrun_data
        
        if not os.path.exists(os.path.join(args.data_dir, "frames.npy")):
            collect_coinrun_data(
                num_levels=config.data.num_levels,
                steps_per_level=config.data.steps_per_level,
                frame_height=config.data.frame_height,
                frame_width=config.data.frame_width,
                data_dir=args.data_dir,
            )
        else:
            print(f"Data already exists at {args.data_dir}/frames.npy, skipping collection")
        
        if args.phase == "collect":
            return
    
    # Phase 2: Train tokenizer
    if args.phase in ["all", "tokenizer"]:
        print("\n" + "=" * 50)
        print("Phase 1: Training Video Tokenizer")
        print("=" * 50)
        
        from genie.train import train_tokenizer
        
        train_tokenizer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_steps=args.tokenizer_steps or config.tokenizer.num_steps,
            batch_size=args.batch_size or config.tokenizer.batch_size,
            device=args.device,
        )
        
        if args.phase == "tokenizer":
            return
    
    # Phase 3: Train LAM + Dynamics
    if args.phase in ["all", "dynamics"]:
        print("\n" + "=" * 50)
        print("Phase 2: Training LAM + Dynamics")
        print("=" * 50)
        
        from genie.train import train_dynamics
        
        tokenizer_path = args.tokenizer_path or os.path.join(args.output_dir, "tokenizer_final.pt")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Run tokenizer phase first or specify --tokenizer_path"
            )
        
        train_dynamics(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            tokenizer_path=tokenizer_path,
            num_steps=args.dynamics_steps or config.dynamics.num_steps,
            batch_size=args.batch_size or config.dynamics.batch_size,
            device=args.device,
        )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Checkpoints saved to: {args.output_dir}")
    print("\nTo play with the trained model:")
    print(f"  python scripts/play.py --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()