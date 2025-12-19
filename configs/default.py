"""
Mini-Genie default configuration.
All hyperparameters in one place for easy tweaking.
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    # CoinRun data collection
    num_levels: int = 1000           # number of different levels to collect from
    steps_per_level: int = 1000      # timesteps per level
    frame_height: int = 64           # resize frames to this height
    frame_width: int = 64            # resize frames to this width
    
    # Dataset
    seq_len: int = 8                 # frames per sequence for training (down from 16 to be memory efficient)
    data_dir: str = "data/coinrun"


@dataclass
class TokenizerConfig:
    # Architecture
    in_channels: int = 3
    hidden_dim: int = 64             # base dimension (scales up: 64 -> 128 -> 256)
    latent_dim: int = 32             # dimension of each codebook vector
    num_codes: int = 1024            # codebook size (vocabulary)
    
    # Spatial compression: 64x64 -> 16x16 tokens (4x downsampling)
    
    # Training
    batch_size: int = 64 # up from 32 to speed up training
    lr: float = 3e-4
    num_steps: int = 50_000
    commitment_cost: float = 0.25    # VQ-VAE commitment loss weight
    
    # Logging
    log_every: int = 500
    save_every: int = 10_000


@dataclass 
class LAMConfig:
    # Architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    num_actions: int = 8             # size of latent action vocabulary
    action_dim: int = 32             # dimension of action embeddings
    
    # Training (shared with dynamics)
    batch_size: int = 64 # up from 32 to speed up training
    lr: float = 3e-4


@dataclass
class DynamicsConfig:
    # Architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    
    # MaskGIT
    mask_ratio_min: float = 0.5      # minimum mask ratio during training
    mask_ratio_max: float = 1.0      # maximum mask ratio during training
    maskgit_steps: int = 8           # iterative decoding steps at inference
    
    # Training
    batch_size: int = 64 # up from 32 to speed up training
    lr: float = 3e-4
    num_steps: int = 50_000
    
    # Logging
    log_every: int = 500
    save_every: int = 10_000


@dataclass
class Config:
    data: DataConfig = None
    tokenizer: TokenizerConfig = None
    lam: LAMConfig = None
    dynamics: DynamicsConfig = None
    
    # General
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "outputs"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig()
        if self.lam is None:
            self.lam = LAMConfig()
        if self.dynamics is None:
            self.dynamics = DynamicsConfig()


def get_config() -> Config:
    """Returns default config. Modify as needed."""
    return Config()