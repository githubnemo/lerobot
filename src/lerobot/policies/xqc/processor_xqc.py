#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
XQC processor — delegates to SAC processor since pre/post processing is identical.
"""

from typing import Any

import torch

from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors
from lerobot.policies.xqc.configuration_xqc import XQCConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_xqc_pre_post_processors(
    config: XQCConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Constructs pre-processor and post-processor pipelines for the XQC policy.

    Identical to SAC's processors since XQC only modifies the network architecture
    and loss computation, not the data preprocessing.

    Args:
        config: The configuration object for the XQC policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    return make_sac_pre_post_processors(config, dataset_stats)
