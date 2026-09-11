"""Base abstractions and protocols for Video Action Models (VAM)."""

from .extractor import (
    BaseExtractorConfig,
    BaseVAMExtractor,
    ExtractorConfigError,
    VAEContractViolationError,
    VAMExtractionOutput,
)
from .lora import (
    BaseLoRALinear,
    BaseVideoLoRAConfig,
    LoRAError,
    extract_lora_state_dict,
    get_lora_parameters,
    inject_base_lora,
    load_lora_weights,
    merge_all_lora_to_base,
    save_lora_weights,
)
from .split_guard import (
    CANONICAL_DATASET_REPO,
    CANONICAL_DATASET_REVISION,
    PROTOCOL_1_0_TRAIN_EPISODES,
    PROTOCOL_1_0_VAL_EPISODES,
    ProtocolSplitGuard,
    ProtocolViolationError,
    enforce_protocol_1_0_split,
    extract_episodes_from_manifest,
    forbid_frame_level_random_split,
    validate_manifests_for_protocol,
)

__all__ = [
    "BaseExtractorConfig",
    "BaseVAMExtractor",
    "ExtractorConfigError",
    "VAEContractViolationError",
    "VAMExtractionOutput",
    "BaseLoRALinear",
    "BaseVideoLoRAConfig",
    "LoRAError",
    "inject_base_lora",
    "get_lora_parameters",
    "extract_lora_state_dict",
    "save_lora_weights",
    "load_lora_weights",
    "merge_all_lora_to_base",
    "PROTOCOL_1_0_TRAIN_EPISODES",
    "PROTOCOL_1_0_VAL_EPISODES",
    "CANONICAL_DATASET_REPO",
    "CANONICAL_DATASET_REVISION",
    "ProtocolViolationError",
    "enforce_protocol_1_0_split",
    "extract_episodes_from_manifest",
    "validate_manifests_for_protocol",
    "forbid_frame_level_random_split",
    "ProtocolSplitGuard",
]
