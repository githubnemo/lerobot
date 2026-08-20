"""Backbone provenance for interchangeable VAM feature artifacts.

The current Cosmos cache schema is deliberately strict and owned by another
change.  This additive schema captures the fields that must be added to its
next version so LTX and Cosmos features cannot be mixed accidentally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BackboneFeatureProvenance:
    """Identity and geometry needed to validate a frozen feature producer."""

    backbone: str
    checkpoint_identity: str
    checkpoint_sha256: str
    hidden_width: int
    layer: int
    noise_parameterization: str
    noise_level: float
    token_geometry: tuple[int, int, int]
    token_count: int
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["token_geometry"] = list(self.token_geometry)
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BackboneFeatureProvenance:
        required = {
            "backbone",
            "checkpoint_identity",
            "checkpoint_sha256",
            "hidden_width",
            "layer",
            "noise_parameterization",
            "noise_level",
            "token_geometry",
            "token_count",
            "dtype",
        }
        if set(payload) != required:
            raise ValueError(
                "backbone provenance keys must match exactly; "
                f"missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}"
            )
        geometry = payload["token_geometry"]
        if (
            not isinstance(geometry, list)
            or len(geometry) != 3
            or any(type(value) is not int or value <= 0 for value in geometry)
        ):
            raise ValueError("token_geometry must be three positive integers")
        for name in ("hidden_width", "layer", "token_count"):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(payload["noise_level"], (int, float)) or payload["noise_level"] < 0:
            raise ValueError("noise_level must be non-negative")
        if any(
            not isinstance(payload[name], str) or not payload[name]
            for name in (
                "backbone",
                "checkpoint_identity",
                "checkpoint_sha256",
                "noise_parameterization",
                "dtype",
            )
        ):
            raise ValueError("backbone identity fields must be non-empty strings")
        return cls(
            backbone=payload["backbone"],
            checkpoint_identity=payload["checkpoint_identity"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            hidden_width=payload["hidden_width"],
            layer=payload["layer"],
            noise_parameterization=payload["noise_parameterization"],
            noise_level=float(payload["noise_level"]),
            token_geometry=tuple(geometry),
            token_count=payload["token_count"],
            dtype=payload["dtype"],
        )


def assert_same_backbone(
    expected: BackboneFeatureProvenance,
    actual: BackboneFeatureProvenance,
) -> None:
    """Reject a cache mix when producer identity or representation changes."""

    if expected != actual:
        raise ValueError(
            f"feature producers are incompatible; expected={expected.to_dict()}, actual={actual.to_dict()}"
        )
