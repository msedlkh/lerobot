# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""

import argparse
import concurrent.futures
import contextlib
import logging
import math
import platform
import queue
import shutil
import time
from pathlib import Path
from threading import Event, Thread
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.utils.robot_utils import (
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

IndexOrPath: TypeAlias = int | Path

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


def is_valid_unix_path(path: str) -> bool:
    """Note: if 'path' points to a symlink, this will return True only if the target exists"""
    p = Path(path)
    return p.is_absolute() and p.exists()


def get_camera_index_from_unix_port(port: Path) -> int:
    return int(str(port.resolve()).removeprefix("/dev/video"))


def save_image(img_array: np.ndarray, camera_index: int, frame_index: int, images_dir: Path):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    camera_idx_or_paths: list[IndexOrPath] | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    record_time_s: int = 2,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given camera index.
    """
    if not camera_idx_or_paths:
        camera_idx_or_paths = OpenCVCamera.find_cameras()
        if len(camera_idx_or_paths) == 0:
            raise RuntimeError(
                "Not a single camera was detected. Try re-plugging, or re-installing `opencv-python`, "
                "or your camera driver, or make sure your camera is compatible with opencv."
            )

    print("Connecting cameras")
    cameras = []
    for idx_or_path in camera_idx_or_paths:
        config = OpenCVCameraConfig(index_or_path=idx_or_path, fps=fps, width=width, height=height)
        camera = OpenCVCamera(config)
        camera.connect()
        print(
            f"OpenCVCamera({camera.index_or_path}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")
    # NOTE(Steven): Cameras don't get disconnected


class OpenCVCamera(Camera):
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
    ```

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

    config = OpenCVCameraConfig(index_or_path=0)
    camera = OpenCVCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = OpenCVCameraConfig(index_or_path=0, fps=30, width=1280, height=720)
    config = OpenCVCameraConfig(index_or_path=0, fps=90, width=640, height=480)
    config = OpenCVCameraConfig(index_or_path=0, fps=90, width=640, height=480, color_mode="bgr")
    # Note: might error out open `camera.connect()` if these settings are not compatible with the camera
    ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.index_or_path = config.index_or_path

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode

        self.camera: cv2.VideoCapture | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        # self.color_image = None # NOTE(Steven): Consider changing this to a Queue?
        self.frame_queue = queue.Queue(maxsize=1)
        self.logs = {}

        self.rotation = get_cv2_rotation(config.rotation)
        self.backend = get_cv2_backend()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self.camera.isOpened() if isinstance(self.camera, cv2.VideoCapture) else False

    def _open_camera(self, index_or_path: IndexOrPath, backend: int) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(index_or_path, backend)

        # If the camera doesn't work, display the camera indices corresponding to valid cameras.
        if not camera.isOpened():
            # Release camera to make it accessible for `find_camera_indices`
            camera.release()
            # Verify that the provided `camera_index` is valid before printing the traceback
            cameras_info = self.find_cameras()
            available_cam_ids = [cam["index"] for cam in cameras_info]
            if index_or_path not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, "
                    f"but {index_or_path} is provided instead. To find the camera index you should use, "
                    "run `python lerobot/common/robot_devices/cameras/opencv.py`."
                )
            raise ConnectionError(f"Can't access {self}.")
        return camera

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Use 1 thread to avoid blocking the main thread. Especially useful during data collection
        # when other threads are used to save the images.
        cv2.setNumThreads(1)

        self.camera = self._open_camera(self.index_or_path, self.backend)

        # NOTE(Steven): What happens if it is none?
        if self.fps is not None:
            self._set_fps(self.fps)
        if self.capture_width is not None:
            self._set_capture_width(self.capture_width)
        if self.capture_height is not None:
            self._set_capture_height(self.capture_height)

    def _set_fps(self, fps: int) -> None:
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if not math.isclose(fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(
                f"Can't set {fps=} for {self}. Actual value is {actual_fps}."
            )  # NOTE(Steven): Consider a more explicit exception? CameraConfigurationError?

    def _set_capture_width(self, capture_width: int) -> None:
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        if not math.isclose(
            self.capture_width, actual_width, rel_tol=1e-3
        ):  # NOTE(Steven): Do we really need isclose()? Couldn't we just cast to int?
            raise RuntimeError(f"Can't set {capture_width=} for {self}. Actual value is {actual_width}.")

    def _set_capture_height(self, capture_height: int) -> None:
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not math.isclose(self.capture_height, actual_height, rel_tol=1e-3):
            raise RuntimeError(f"Can't set {capture_height=} for {self}. Actual value is {actual_height}.")

    @staticmethod
    def find_cameras(max_index_search_range=MAX_OPENCV_INDEX) -> list[IndexOrPath]:
        if platform.system() == "Linux":
            print("Linux detected. Finding available camera indices through scanning '/dev/video*' ports")
            possible_idx_or_paths = [str(port) for port in Path("/dev").glob("video*")]
        else:
            print(
                f"{platform.system()} system detected. Finding available camera indices through "
                f"scanning all indices from 0 to {MAX_OPENCV_INDEX}"
            )
            possible_idx_or_paths = range(max_index_search_range)

        found_idx_or_paths = []
        for target in possible_idx_or_paths:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                print(f"Camera found at {target}")
                found_idx_or_paths.append(target)
                camera.release()

        return found_idx_or_paths

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        ret, color_image = self.camera.read()
        if not ret:
            raise RuntimeError(f"Can't capture color image from {self}.")

        color_image = self._postprocess_image(color_image, color_mode)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read: {dt_ms:.1f}ms")

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        return color_image

    def _postprocess_image(self, image, color_mode: ColorMode | None = None):
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):  # NOTE(Steven): Use new enums?
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == ColorMode.RGB:
            color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE(Steven): I think this is better placed in read() if not used in async
        h, w, _ = color_image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). "
                f"({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        return color_image

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                color_image = self.read()
                with contextlib.suppress(queue.Empty):
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put(color_image)
            except Exception as e:
                # NOTE(Steven): Consider logging the error instead of printing
                print(f"Error reading in thread: {e}")
                # NOTE(Steven): Consider small sleep here to avoid spam

    def async_read(self, timeout_ms: float = 2000):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            if self.thread is not None:
                self.stop_event.set()
                self.thread.join(timeout=0.5)
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.stop_event = Event()
            self.thread = Thread(
                target=self._read_loop, args=(), name=f"OpenCVCameraReadLoop-{self.index_or_path}"
            )
            self.thread.daemon = True
            self.thread.start()
            init_wait = min(0.5, 2 / self.fps if self.fps and self.fps > 0 else 0.1)
            time.sleep(init_wait)

        try:
            # NOTE(Steven): No postprocessing here?
            return self.frame_queue.get(timeout=timeout_ms / 1000)

        except queue.Empty:
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self.index_or_path} after {timeout_ms} milliseconds. "
                f"(Read thread alive: {thread_alive})"
            ) from queue.Empty
        except Exception as e:
            raise RuntimeError(f"Error getting frame from queue for camera {self.index_or_path}: {e}") from e

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # wait for the thread to finish # NOTE(Steven): Consider timeout + check status?
            self.thread = None
            self.stop_event = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        logger.debug(f"Camera {self.index_or_path} disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
