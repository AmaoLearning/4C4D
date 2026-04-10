"""Lazy-loading DataLoader utilities for double-buffer GPU image pipeline.

Provides:
- ``CameraImageDataset``: Dataset that decodes images in worker processes
  and returns ``(cam_index, image_tensor)`` with pinned CPU memory.
- ``create_camera_dataloader``: Factory that wires up DataLoader with
  pin_memory, persistent_workers, and shuffle.
- ``InfiniteDataLoader``: Wrapper that auto-restarts the iterator so
  ``next()`` never raises ``StopIteration``.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CameraImageDataset(Dataset):
    """Worker-side image decoder for the lazy-loading pipeline.

    Each ``__getitem__`` call reads an image from disk (via cv2), resizes
    to the camera's target resolution, normalises to ``[0, 1]`` float32,
    and returns ``(cam_index, image_tensor [3, H, W])``.

    The returned tensor is a regular CPU tensor; ``pin_memory=True`` on
    the DataLoader will pin it automatically for fast async DMA copies.
    """

    def __init__(self, cameras, white_background: bool = False):
        self.cameras = cameras
        self.bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, index):
        cam = self.cameras[index]
        img = cv2.imread(cam.image_path, cv2.IMREAD_UNCHANGED)

        target_w, target_h = cam.resolution
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0

        if img.ndim == 3 and img.shape[2] == 4:
            rgb = img[:, :, 2::-1]   # BGR → RGB
            alpha = img[:, :, 3:4]
            blended = rgb * alpha + self.bg * (1.0 - alpha)
        else:
            blended = img[:, :, 2::-1]  # BGR → RGB

        # [H, W, 3] → [3, H, W], contiguous for fast DMA
        image_tensor = torch.from_numpy(blended.copy()).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
        return index, image_tensor


def create_camera_dataloader(
    cameras,
    white_background: bool = False,
    batch_size: int = 1,
    num_workers: int = 16,
    prefetch_factor: int = 12,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    shuffle: bool = True,
):
    """Create a DataLoader for lazy image loading with pinned memory."""
    dataset = CameraImageDataset(cameras, white_background=white_background)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )
    return dl


class InfiniteDataLoader:
    """Wraps a DataLoader to cycle forever — ``next()`` never raises."""

    def __init__(self, dataloader: DataLoader):
        self._dl = dataloader
        self._iter = iter(self._dl)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._dl)
            return next(self._iter)
