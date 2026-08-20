"""Reusable dataset contracts for video-conditioned action modeling."""

from .config import CUBE_OUT_OF_BOX_CONTRACT, VideoVAMDatasetConfig
from .validation import (
    DatasetContractError,
    ValidationReport,
    validate_dataset,
    validate_metadata,
    validate_sample,
    validate_samples,
)

__all__ = [
    "CUBE_OUT_OF_BOX_CONTRACT",
    "DatasetContractError",
    "ValidationReport",
    "VideoVAMDatasetConfig",
    "validate_dataset",
    "validate_metadata",
    "validate_sample",
    "validate_samples",
]
