"""Decode selected RGB frames from Franka NPZ or ALOHA HDF5 recordings."""

import io
from pathlib import Path

import numpy as np


def validate_timing(times, steps, frames, path):
    if (times.shape != (frames,) or steps.shape != (frames,)
            or not np.isfinite(times).all() or not np.all(np.diff(steps) == 5)
            or not np.allclose(np.diff(times), 0.02, atol=1e-5)
            or not np.allclose(times, steps * 0.004, atol=1e-5)):
        raise ValueError(f"Invalid 50 Hz recording time axis: {path}")


def read_rgb(path, expected_frames, indices, views=1):
    """Return N×views×224×224×3 RGB in external, left-wrist order."""
    path = Path(path)
    if views not in (1, 2):
        raise ValueError("Expected one or two camera views")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= expected_frames:
        raise ValueError(f"Frame indices outside recording: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            rgb = data["rgb"]
            validate_timing(data["timestamp_s"], data["observation_step"], expected_frames, path)
        if rgb.shape != (expected_frames, 2, 224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Unexpected Franka RGB shape or dtype: {path}")
        return np.ascontiguousarray(rgb[indices, :views])
    if path.suffix != ".hdf5":
        raise ValueError(f"Unsupported recording container: {path}")
    if views != 1:
        raise ValueError("ALOHA demonstrations use external RGB only")
    import h5py
    from PIL import Image

    images = []
    with h5py.File(path, "r") as data:
        validate_timing(data["timestamp"][:], data["physics_step"][:], expected_frames, path)
        camera = data["observation/head_camera/rgb"]
        if camera.shape != (expected_frames,):
            raise ValueError(f"Unexpected ALOHA head stream: {path}")
        for index in indices:
            with Image.open(io.BytesIO(bytes(camera[int(index)]))) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape != (224, 224, 3):
                raise ValueError(f"Unexpected ALOHA head image size: {path}")
            images.append(rgb)
    return np.stack(images)[:, None]
