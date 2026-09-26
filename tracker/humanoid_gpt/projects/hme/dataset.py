import os
import tqdm
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def compute_win_len(win_sec: float, downsample_rate: int, frequency: float = 50) -> int:
    """Compute window length from duration, frequency and downsample rate."""
    return int(win_sec * frequency / downsample_rate) + 1


class PAEDataset(Dataset):
    def __init__(
            self,
            npz_path: str,
            win_sec: float = 2.0,
            downsample_rate: int = 5,
            stride: int = 1,
            verbose: bool = True,
    ):
        """
        Lazy-loading dataset for PAE training.
        
        Args:
            npz_path: path to directory containing npz files
            win_sec: window duration in seconds
            downsample_rate: temporal downsampling rate
            stride: skip files (use every stride-th file)
            verbose: whether to print progress
        """
        self.win_sec = win_sec
        self.downsample_rate = downsample_rate
        
        # Scan files and compute valid sample counts
        self.file_info = []  # [(path, frequency, num_samples), ...]
        self.cumsum = [0]    # cumulative sample count for indexing
        
        npz_files = sorted(Path(npz_path).rglob("*.npz"))[::stride]
        iterator = tqdm.tqdm(npz_files, desc="Scanning files") if verbose else npz_files
        for npz in iterator:
            try:
                # Only load metadata, not full data
                data = np.load(npz, mmap_mode='r')
                trj_len = data["qpos"].shape[0]
                frequency = float(data["frequency"]) if "frequency" in data else 50
            except Exception as e:
                if verbose:
                    print(f"Error loading {npz}: {e}")
                continue
            
            win_len = compute_win_len(win_sec, downsample_rate, frequency)
            win_len_half = (win_len - 1) // 2
            padding = win_len_half * downsample_rate
            num_samples = trj_len - 2 * padding
            
            if num_samples > 0:
                self.file_info.append((str(npz), frequency, num_samples))
                self.cumsum.append(self.cumsum[-1] + num_samples)
        
        assert len(self.file_info) > 0, "No valid motion files found."
        if verbose:
            print(f"Found {len(self.file_info)} files, {self.cumsum[-1]} total samples")
    
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, idx):
        # Binary search to find which file this idx belongs to
        file_idx = np.searchsorted(self.cumsum[1:], idx, side='right')
        local_idx = idx - self.cumsum[file_idx]
        
        path, frequency, _ = self.file_info[file_idx]
        
        # Load data
        data = np.load(path)
        qpos = torch.from_numpy(data["qpos"].astype(np.float32))
        qvel = torch.from_numpy(data["qvel"].astype(np.float32))
        gv_vel = torch.from_numpy(data["gv_vel"].astype(np.float32))
        
        # Concatenate features
        features = torch.cat([qpos, qvel, gv_vel], dim=-1)  # (N, 74)
        
        # Compute window parameters
        win_len = compute_win_len(self.win_sec, self.downsample_rate, frequency)
        win_len_half = (win_len - 1) // 2
        padding = win_len_half * self.downsample_rate
        
        # Extract window at local_idx
        center = padding + local_idx
        win_offset = torch.arange(-win_len_half, win_len_half + 1) * self.downsample_rate
        win_ids = center + win_offset
        
        return features[win_ids]  # (win_len, 74)
