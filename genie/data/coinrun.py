"""
CoinRun data collection and dataset.

Collects frames from CoinRun environment using random policy.
CoinRun is a procedurally generated 2D platformer — ideal for 
learning latent actions like left/right/jump.

Note: procgen requires Python 3.7-3.10 and Linux.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def collect_coinrun_data(
    num_levels: int = 1000,
    steps_per_level: int = 1000,
    frame_height: int = 64,
    frame_width: int = 64,
    data_dir: str = "data/coinrun",
):
    """
    Collect frames from CoinRun using random actions.
    
    Args:
        num_levels: Number of different levels to collect from
        steps_per_level: Timesteps to collect per level
        frame_height: Resize frames to this height
        frame_width: Resize frames to this width
        data_dir: Directory to save frames
    
    Saves frames as .npy file: (total_frames, H, W, 3) uint8
    """
    # Import here since procgen only works on Linux
    try:
        from procgen import ProcgenEnv
    except ImportError:
        raise ImportError(
            "procgen not installed. Run on Linux with Python ≤3.10:\n"
            "  pip install procgen"
        )
    
    os.makedirs(data_dir, exist_ok=True)
    
    all_frames = []
    
    print(f"Collecting {num_levels} levels × {steps_per_level} steps = {num_levels * steps_per_level} frames")
    
    for level_seed in tqdm(range(num_levels), desc="Collecting levels"):
        # Create environment for this level
        env = ProcgenEnv(
            num_envs=1,
            env_name="coinrun",
            start_level=level_seed,
            num_levels=1,  # Just this one level
            distribution_mode="hard",
        )
        
        obs = env.reset()
        
        for step in range(steps_per_level):
            # Random action
            action = np.array([env.action_space.sample()])
            obs, rewards, dones, infos = env.step(action)
            
            # Get frame (procgen returns dict with 'rgb' key)
            frame = obs['rgb'][0]  # (64, 64, 3) uint8
            
            # Resize if needed
            if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((frame_width, frame_height), Image.BILINEAR)
                frame = np.array(pil_img)
            
            all_frames.append(frame)
            
            # Reset if done
            if dones[0]:
                obs = env.reset()
        
        env.close()
    
    # Stack and save
    all_frames = np.stack(all_frames, axis=0)  # (N, H, W, 3)
    
    save_path = os.path.join(data_dir, "frames.npy")
    np.save(save_path, all_frames)
    
    print(f"Saved {len(all_frames)} frames to {save_path}")
    print(f"Shape: {all_frames.shape}, Size: {all_frames.nbytes / 1e9:.2f} GB")
    
    return save_path


class CoinRunDataset(Dataset):
    """
    Dataset for training video tokenizer and dynamics model.
    
    Returns sequences of consecutive frames.
    For tokenizer: just need individual frames (seq_len=1 is fine)
    For dynamics: need sequences to learn temporal patterns
    """
    
    def __init__(
        self,
        data_dir: str = "data/coinrun",
        seq_len: int = 8,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        """
        Args:
            data_dir: Directory containing frames.npy
            seq_len: Number of consecutive frames per sample
            split: "train" or "val"
            train_ratio: Fraction of data for training
        """
        self.seq_len = seq_len
        
        # Load frames
        frames_path = os.path.join(data_dir, "frames.npy")
        if not os.path.exists(frames_path):
            raise FileNotFoundError(
                f"No data found at {frames_path}. Run collect_coinrun_data() first."
            )
        
        self.frames = np.load(frames_path)  # (N, H, W, 3) uint8
        
        # Split into train/val
        # We split by "sequence start points" to avoid train/val overlap
        num_frames = len(self.frames)
        num_sequences = num_frames - seq_len + 1
        
        split_idx = int(num_sequences * train_ratio)
        
        if split == "train":
            self.start_indices = list(range(0, split_idx))
        else:
            self.start_indices = list(range(split_idx, num_sequences))
        
        print(f"CoinRunDataset ({split}): {len(self.start_indices)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.start_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            frames: (seq_len, 3, H, W) float32 tensor, values in [0, 1]
        """
        start = self.start_indices[idx]
        frames = self.frames[start : start + self.seq_len]  # (seq_len, H, W, 3)
        
        # Convert to tensor: (seq_len, H, W, 3) -> (seq_len, 3, H, W)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (seq_len, 3, H, W)
        
        return frames


class SingleFrameDataset(Dataset):
    """
    Simpler dataset that returns individual frames.
    Use this for training the video tokenizer.
    """
    
    def __init__(
        self,
        data_dir: str = "data/coinrun",
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        frames_path = os.path.join(data_dir, "frames.npy")
        if not os.path.exists(frames_path):
            raise FileNotFoundError(
                f"No data found at {frames_path}. Run collect_coinrun_data() first."
            )
        
        self.frames = np.load(frames_path)  # (N, H, W, 3) uint8
        
        # Split
        num_frames = len(self.frames)
        split_idx = int(num_frames * train_ratio)
        
        if split == "train":
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, num_frames))
        
        print(f"SingleFrameDataset ({split}): {len(self.indices)} frames")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            frame: (3, H, W) float32 tensor, values in [0, 1]
        """
        frame = self.frames[self.indices[idx]]  # (H, W, 3)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # (3, H, W)
        return frame


# Quick test with dummy data
if __name__ == "__main__":
    import tempfile
    
    # Create dummy data for testing (since we can't run procgen locally)
    print("Creating dummy data for testing...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake frames
        dummy_frames = np.random.randint(0, 255, (1000, 64, 64, 3), dtype=np.uint8)
        np.save(os.path.join(tmpdir, "frames.npy"), dummy_frames)
        
        # Test SingleFrameDataset
        dataset = SingleFrameDataset(data_dir=tmpdir, split="train")
        frame = dataset[0]
        print(f"SingleFrame shape: {frame.shape}, range: [{frame.min():.2f}, {frame.max():.2f}]")
        
        # Test CoinRunDataset
        seq_dataset = CoinRunDataset(data_dir=tmpdir, seq_len=8, split="train")
        seq = seq_dataset[0]
        print(f"Sequence shape: {seq.shape}, range: [{seq.min():.2f}, {seq.max():.2f}]")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(seq_dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")  # Should be (4, 8, 3, 64, 64)