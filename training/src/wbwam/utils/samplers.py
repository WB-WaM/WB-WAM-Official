from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.resume_batch_offset = 0

        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {self.batch_size}.")
        if self.num_replicas <= 0:
            raise ValueError(f"`num_replicas` must be > 0, got {self.num_replicas}.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"`rank` must be in [0, {self.num_replicas}), got {self.rank}."
            )
        if len(self.dataset) <= 0:
            raise ValueError("`dataset` must be non-empty for ResumableDistributedSampler.")
        if self.drop_last and len(self.dataset) < self.num_replicas:
            raise ValueError(
                "`dataset` length must be >= num_replicas when drop_last=True, "
                f"got len={len(self.dataset)} num_replicas={self.num_replicas}."
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)
        if self.resume_batch_offset < 0:
            raise ValueError(
                f"`resume_batch_offset` must be >= 0, got {self.resume_batch_offset}."
            )

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    @property
    def num_samples(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.num_replicas
        return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        total_size = self.num_samples * self.num_replicas
        if self.drop_last:
            indices = indices[:total_size]
        else:
            padding_size = total_size - len(indices)
            if padding_size > 0:
                repeats = (padding_size + len(indices) - 1) // len(indices)
                indices += (indices * repeats)[:padding_size]
            else:
                indices = indices[:total_size]

        indices = indices[self.rank:total_size:self.num_replicas]
        if self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size
            indices = indices[sample_offset:]
            self.resume_batch_offset = 0
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
