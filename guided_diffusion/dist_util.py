"""Device and checkpoint helpers for standalone single-process inference."""

from pathlib import Path

import torch as th


def setup_dist():
    """Retained for API compatibility; standalone inference is single-process."""
    return None


def dev():
    """Return the CUDA device allocated by Slurm, or CPU when CUDA is absent."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")


def load_state_dict(path, **kwargs):
    """Load one local PyTorch checkpoint."""
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return th.load(str(checkpoint), **kwargs)


def sync_params(params):
    """No-op retained for compatibility with code shared with training."""
    return None
