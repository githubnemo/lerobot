# LeRobot-owned compatibility seam for the copied Cosmos source.
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


class _CosmosLoggerAdapter:
    """Expose imaginaire's success method without changing stdlib logging."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def success(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Map upstream success messages to the standard info level."""
        self._logger.info(message, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Preserve standard logger methods and attributes."""
        return getattr(self._logger, name)


log = _CosmosLoggerAdapter(logging.getLogger("lerobot.policies.vam.cosmos_predict2"))


class DataType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CosmosTextEncoderConfig:
    """Shape-only replacement; T5 is loaded outside this extractor."""

    NUM_TOKENS: int = 512
    EMBED_DIM: int = 1024


class _EasyIO:
    @staticmethod
    def load(path: str, *args: Any, map_location: str | torch.device = "cpu", **kwargs: Any) -> Any:
        del args, kwargs
        return torch.load(path, map_location=map_location, weights_only=False)


easy_io = _EasyIO()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def sync_model_states(model: torch.nn.Module) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor, src=0)


def broadcast(item: Any, src: int = 0, process_group: Any = None) -> Any:
    if process_group is None or not dist.is_available() or not dist.is_initialized():
        return item
    if isinstance(item, torch.Tensor):
        dist.broadcast(item, src=src, group=process_group)
    return item


def create_cuda_graph(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("CUDA graph capture is outside the LeRobot Cosmos extractor foundation")


# This namespace is only used for a rank query. It never initializes distributed state.
distributed = SimpleNamespace(get_rank=get_rank)
