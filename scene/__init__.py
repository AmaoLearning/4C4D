#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import json
from typing import Optional, List, Tuple
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, 
                 resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, 
                 training_view=['cam10', 'cam01', 'cam20', 'cam13'], redundant_ratio=0.2,
                 downsample_method='random', testing_view=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        # ── Lazy loading state ──
        self._lazy_mode: bool = args.lazy_load
        self._load2gpu_on_the_fly: bool = args.load2gpu_on_the_fly
        self._lazy_num_workers: int = int(args.lazy_num_workers)
        self._lazy_prefetch_factor: int = int(args.lazy_prefetch_factor)
        self._lazy_image_buffer_count: int = max(2, int(args.lazy_image_buffer_count))
        self._lazy_data_iter = None
        self._gpu_image_buffers: Optional[List[torch.Tensor]] = None
        self._lazy_cameras: Optional[List] = None
        self._active_buf_idx: int = 0
        self._dma_stream: Optional[torch.cuda.Stream] = None
        self._prefetch_ready: Optional[Tuple[int, int, torch.cuda.Event]] = None
        self._viewpoint_stack: list = []  # for non-lazy random sampling

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        else:
            print("Creating new model")
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print(f"Found poses_bounds.npy in {args.source_path}, assuming Dynerf/N3V data set!")
            _hold_id = [int(args.dynerf_hold_id)]
            scene_info = sceneLoadTypeCallbacks["dynerf"](
                args.source_path, args.white_background, args.eval,
                num_images=args.num_images, hold_id=_hold_id,
                dataloader=args.dataloader,
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio, 
                                                          training_cam=training_view, testing_cam=testing_view, num_pts=num_pts, time_duration=time_duration,
                                                          downsample_method=downsample_method)
            print(f"Found sparse folder in {args.source_path}, assuming Colmap data set!")
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print(f"Found transforms_train.json file in {args.source_path}, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                print(f"Copying input.ply from source path: {args.source_path} to model path: {self.model_path}.")
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                print(f"Writing cameras.json to model path: {self.model_path}.")
                json.dump(json_cams, file)

        if shuffle:
            print("Shuffling training and testing cameras")
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            print("Shuffling done.")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print(f"Loading cameras at resolution scale {resolution_scale}")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loaded Training Cameras with {len(self.train_cameras[resolution_scale])} frames with resolution scale {resolution_scale}")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loaded Testing Cameras with {len(self.test_cameras[resolution_scale])} frames with resolution scale {resolution_scale}")
            
        if args.loaded_pth:
            print(f"Loading gaussians from {args.loaded_pth}")
            self.gaussians.create_from_pth(args.loaded_pth, self.cameras_extent)
        else:
            if self.loaded_iter:
                print(f"Loading gaussians from trained model's point cloud at iteration {self.loaded_iter}")
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                print(f"Creating gaussians from initial point cloud input.ply")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                                               redundant_ratio=redundant_ratio)
                print(f"Created gaussians from input.ply")

    def save(self, iteration):
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
    
    def getValidationCameras(self, scale=1.0, tag='train', num=100):
        if tag == 'train':
            return CameraDataset(self.train_cameras[scale][::num], self.white_background)
        elif tag == 'test':
            return CameraDataset(self.test_cameras[scale][::num], self.white_background)
        
    def getAllCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy() + self.test_cameras[scale].copy(), self.white_background)

    # ── Lazy loading API ─────────────────────────────────────────────

    @property
    def lazy_mode(self) -> bool:
        return self._lazy_mode

    def setup_lazy_dataloader(
        self,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        scale: float = 1.0,
    ) -> None:
        """Initialize the async image prefetch pipeline.

        Call once after ``__init__``.  No-op if ``lazy_load`` is disabled.
        When all cameras share the same resolution, a persistent GPU buffer
        is pre-allocated so that training iterations incur **zero**
        ``cudaMalloc`` / ``cudaFree``.
        """
        from utils.dataload_utils import create_camera_dataloader, InfiniteDataLoader

        if not self._lazy_mode:
            return

        if num_workers is None:
            num_workers = self._lazy_num_workers
        if prefetch_factor is None:
            prefetch_factor = self._lazy_prefetch_factor

        cameras = self.train_cameras.get(scale, [])
        if not cameras:
            print("[WARN] No training cameras found for lazy dataloader")
            return
        self._lazy_cameras = cameras

        _dl = create_camera_dataloader(
            cameras,
            white_background=self.white_background,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )
        self._lazy_data_iter = InfiniteDataLoader(_dl)

        # Pre-allocate fixed CUDA buffers if every camera uses the same
        # resolution (the common case for N3V / COLMAP with uniform res).
        resolutions = set(c.resolution for c in cameras)
        if len(resolutions) == 1:
            w, h = resolutions.pop()
            self._gpu_image_buffers = [
                torch.empty(3, h, w, dtype=torch.float32, device="cuda")
                for _ in range(self._lazy_image_buffer_count)
            ]
            self._dma_stream = torch.cuda.Stream()
            # Pre-fill: DMA the first batch so the first
            # next_train_camera() call never stalls on a cold buffer.
            self._kick_prefetch()
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"GPU image-buffers={len(self._gpu_image_buffers)}x{w}x{h} "
                  f"(zero-alloc, DMA overlap)")
        else:
            print(f"[INFO] Lazy DataLoader: {len(cameras)} cameras, "
                  f"{num_workers} workers, prefetch={prefetch_factor}, "
                  f"(mixed resolutions, no persistent buffer)")

    def _kick_prefetch(
        self, wait_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Start async DMA of the next DataLoader batch into the inactive
        GPU buffer on ``self._dma_stream``.

        Called at the END of ``next_train_camera()`` so the transfer runs
        concurrently with render + backward on the default stream.
        """
        _batch_idx, _batch_img = next(self._lazy_data_iter)
        cam_idx = _batch_idx.item()
        pinned_img = _batch_img.squeeze(0)  # [3, H, W] pinned CPU

        next_buf = (self._active_buf_idx + 1) % len(self._gpu_image_buffers)
        with torch.cuda.stream(self._dma_stream):
            if wait_event is not None:
                self._dma_stream.wait_event(wait_event)
            self._gpu_image_buffers[next_buf].copy_(
                pinned_img, non_blocking=True)

        self._prefetch_ready = (
            cam_idx,
            next_buf,
            self._dma_stream.record_event(),
        )

    def next_train_camera(self, scale: float = 1.0):
        """Return ``(camera, gt_image_on_gpu)`` with the image ready on GPU.

        * **Lazy mode** — pulls from the async DataLoader and copies the
          pre-fetched pinned tensor into the persistent GPU buffer (or
          falls back to ``.cuda()`` for mixed-resolution datasets).
        * **Eager mode** — pops a random camera from an internal
          ``viewpoint_stack`` that auto-refills each epoch.
        """
        if self._lazy_mode:
            assert self._lazy_data_iter is not None, (
                "Call scene.setup_lazy_dataloader() before next_train_camera()")
            cameras = self._lazy_cameras

            if (self._gpu_image_buffers is not None
                    and self._prefetch_ready is not None):
                # ── Double-buffer fast path ──
                done_event = torch.cuda.Event()
                done_event.record()  # mark current default-stream work done

                cam_idx, buf_idx, dma_event = self._prefetch_ready
                self._prefetch_ready = None

                # Wait for the DMA copy to finish on the default stream
                torch.cuda.current_stream().wait_event(dma_event)

                cam = cameras[cam_idx]
                gt_image = self._gpu_image_buffers[buf_idx]

                # Inject the GPU image into the camera for compatibility
                cam.image = gt_image
                self._active_buf_idx = buf_idx

                # Kick the next prefetch (runs on DMA stream, overlaps
                # with render + backward on the default stream)
                self._kick_prefetch(wait_event=done_event)

                return cam, gt_image
            else:
                # Fallback for mixed-resolution: sync load
                _batch_idx, _batch_img = next(self._lazy_data_iter)
                cam_idx = _batch_idx.item()
                cam = cameras[cam_idx]
                gt_image = _batch_img.squeeze(0).cuda()
                cam.image = gt_image
                return cam, gt_image

        # ── Eager (non-lazy) path ──
        if not self._viewpoint_stack:
            self._viewpoint_stack = self.train_cameras.get(scale, []).copy()

        idx = random.randint(0, len(self._viewpoint_stack) - 1)
        cam = self._viewpoint_stack.pop(idx)

        if self._load2gpu_on_the_fly:
            cam.image = cam.image.cuda()

        return cam, cam.image

    def release_camera_image(self, cam) -> None:
        """Drop the camera's image reference after ``loss.backward()``.

        In lazy-buffer mode the underlying GPU memory stays allocated on
        ``self._gpu_image_buffers`` — only the Python reference is cleared,
        so no ``cudaFree`` ever occurs.
        """
        if self._lazy_mode:
            cam.image = None  # buffer stays alive on self
        elif self._load2gpu_on_the_fly:
            cam.image = cam.image.cpu() if cam.image is not None else None
    