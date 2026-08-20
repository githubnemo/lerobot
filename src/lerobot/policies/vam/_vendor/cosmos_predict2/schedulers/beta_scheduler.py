# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class BetaScheduler:
    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1,
        num_denoising_steps: int = 10,
    ):
        self.alpha = alpha
        self.beta = beta

        self.num_denoising_steps = num_denoising_steps
        self.dt = -1.0 / self.num_denoising_steps

    def sample_t(self, batch_size: int) -> torch.Tensor:
        gamma_base = torch.full((batch_size,), self.alpha, device="cuda", dtype=torch.float32)
        gamma1 = torch._standard_gamma(gamma_base)
        gamma_base.fill_(self.beta)
        gamma2 = torch._standard_gamma(gamma_base)

        return gamma1 / (gamma1 + gamma2) * 0.999 + 0.001

    def step(
        self,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ):
        """The simple euler integration from pi0.

        Parameters
        ----------
        v_t : torch.Tensor
            Prediction of the denoising field at the sample.
        x_t : torch.Tensor
            Current noisy sample x_t.
        """
        prev_xt_dtype = x_t.dtype
        prev_time_dtype = time.dtype

        v_t = v_t.to(torch.float64)
        x_t = x_t.to(torch.float64)
        time = time.to(torch.float64)

        x_t = x_t + self.dt * v_t
        time = time + self.dt

        return x_t.to(prev_xt_dtype), time.to(prev_time_dtype)
