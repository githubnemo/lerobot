#!/usr/bin/env python

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
import dataclasses
import logging
import math
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import copy

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from tqdm import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def get_default_peft_configuration(policy_type):
    if policy_type == "smolvla":
        return {
            "target_modules": r"(model\.vlm_with_expert\.lm_expert\..*\.(q_proj|v_proj)|model\.action_.*|model\.state_proj.*)",
            "modules_to_save": [
                # These are inf on load otherwise
                "normalize_inputs",
                "normalize_targets",
                "unnormalize_outputs",
            ],
        }

    return {'modules_to_save': None}


def wrap_policy_in_peft_model(cfg, policy):
    from peft import get_peft_model, PEFT_TYPE_TO_CONFIG_MAPPING, PeftType

    # Disable all gradients because we'll only train the parameters selected by the PEFT method.
    # Layers that should receive gradients anyway need to be listed in `modules_to_save`.
    for p in policy.parameters():
        p.requires_grad_(False)

    peft_config_policy = get_default_peft_configuration(cfg.policy.type)
    peft_config_cli = dataclasses.asdict(cfg.peft) if cfg.peft else {}
    peft_method_type = PeftType[peft_config_cli["method_type"].upper()]
    peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

    # Handle specific CLI overrides
    for key in ["target_modules", "modules_to_save", "r"]:
        if peft_config_cli[key] is not None:
            peft_config_policy[key] = peft_config_cli[key]

    if 'target_modules' not in peft_config_policy:
        raise ValueError(
            f"There is no default `target_modules` value for policy {cfg.policy.type}. Please pass it manually."
        )

    # Init method depends on the used PEFT method, your specific PEFT method
    # might not be considered here, in that case an error is raised.
    if peft_config_cli["init_type"] is not None:
        if peft_method_type == "LORA":
            peft_config_policy["init_lora_weights"] = peft_config_cli["init_type"]
        elif peft_method_type == "BONE":
            peft_config_policy["init_weights"] = peft_config_cli["init_type"]
        else:
            raise ValueError(
                f"Init type {peft_config_cli['init_type']} unknown for PEFT method {peft_method_type}."
            )

    policy = get_peft_model(
        policy,
        peft_config_cls(**peft_config_policy),
    )

    return policy


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # Set TOKENIZERS_PARALLELISM to false to avoid warnings when forking processes for data loading.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset(s)")
    if cfg.use_validation:
        # Create train dataset with augmentations
        train_dataset = make_dataset(cfg)

        # Create validation dataset without augmentations
        val_cfg = copy.deepcopy(cfg)
        val_cfg.dataset.image_transforms.enable = False
        val_dataset = make_dataset(val_cfg)

        train_indices, val_indices, num_train_episodes, num_val_episodes = create_train_val_splits(cfg, train_dataset)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        dataset = train_dataset
        num_frames = len(train_indices)
        num_episodes = num_train_episodes

    else:
        dataset = make_dataset(cfg)
        val_dataloader = None
        train_indices, val_indices, num_episodes, _ = create_train_val_splits(cfg, dataset)
        num_frames = len(train_indices)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    ds_meta = dataset.dataset.meta if isinstance(dataset, Subset) else dataset.meta
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )

    if cfg.use_peft:
        logging.info("Using PEFT! Wrapping model.")
        policy = wrap_policy_in_peft_model(cfg, policy)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{num_frames=} ({format_big_number(num_frames)})")
    logging.info(f"{num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, num_frames, num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    pbar = tqdm(range(step, cfg.steps), desc="Training")
    latest_val_loss = None  # Track latest validation loss for progress bar
    for _ in pbar:
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()

        # Update progress bar with current loss and validation loss if available
        postfix_dict = {"loss": f"{train_tracker.loss.avg:.2e}"}
        if latest_val_loss is not None:
            postfix_dict["val_loss"] = f"{latest_val_loss:.2e}"
        pbar.set_postfix(postfix_dict)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = cfg.use_validation and cfg.val_freq > 0 and step % cfg.val_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if is_val_step and val_dataloader is not None:
            start_time = time.perf_counter()
            val_results = compute_validation_loss(
                policy,
                val_dataloader,
                device,
                use_amp=cfg.policy.use_amp,
            )
            val_time = time.perf_counter() - start_time

            val_loss = val_results["val_loss"]
            val_n_samples = val_results["val_n_samples"]
            latest_val_loss = val_loss  # Update for progress bar display

            logging.info(
                f"Validation - Step {step}: loss={val_loss:.4f}, samples={val_n_samples}, time={val_time:.2f}s"
            )

            if wandb_logger:
                wandb_log_dict = {
                    "val_loss": val_loss,
                    "val_n_samples": val_n_samples,
                    "val_time_s": val_time,
                }
                wandb_logger.log_dict(wandb_log_dict, step)

            # Return policy to training mode
            policy.train()

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, num_frames, num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def create_train_val_splits(
    cfg: TrainPipelineConfig, dataset: LeRobotDataset
) -> tuple[list[int], list[int] | None, int, int | None]:
    """Create train/validation index splits from the full dataset.

    This split is deterministic and reproducible if a seed is provided in the config:
    - Always splits at episode boundaries (maintains episode integrity)
    - Uses the same episodes for train/val across runs with the same val_split and seed.
    - If no seed is provided, the split will be random.

    Returns:
        A tuple containing:
        - train_indices (list[int]): Frame indices for the training set.
        - val_indices (list[int] | None): Frame indices for the validation set.
        - num_train_episodes (int): Number of episodes in the training set.
        - num_val_episodes (int | None): Number of episodes in the validation set.
    """
    episode_data_index = dataset.episode_data_index
    total_episodes = len(episode_data_index["from"])

    if not cfg.use_validation:
        return list(range(len(dataset))), None, total_episodes, None

    # Calculate indices for train/val split based on episodes
    # Create a list of episode indices
    all_episode_indices = list(range(total_episodes))

    # Shuffle episodes for random split, but seeded for reproducibility
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_episode_indices)

    num_val_episodes = math.ceil(total_episodes * cfg.val_split)
    num_train_episodes = total_episodes - num_val_episodes

    val_ep_indices = all_episode_indices[:num_val_episodes]
    train_ep_indices = all_episode_indices[num_val_episodes:]

    # Get frame indices for training episodes
    train_indices = [
        i
        for ep_idx in train_ep_indices
        for i in range(episode_data_index["from"][ep_idx].item(), episode_data_index["to"][ep_idx].item())
    ]

    # Get frame indices for validation episodes
    val_indices = [
        i
        for ep_idx in val_ep_indices
        for i in range(episode_data_index["from"][ep_idx].item(), episode_data_index["to"][ep_idx].item())
    ]

    logging.info(f"Dataset split - Total episodes: {total_episodes}")
    logging.info(f"Training episodes: {num_train_episodes} ({100 * num_train_episodes / total_episodes:.1f}%)")
    logging.info(f"Validation episodes: {num_val_episodes} ({100 * num_val_episodes / total_episodes:.1f}%)")

    avg_len_train = len(train_indices) / num_train_episodes if num_train_episodes > 0 else 0
    logging.info(f"Training frames: {len(train_indices)} (avg length: {avg_len_train:.1f})")

    avg_len_val = len(val_indices) / num_val_episodes if num_val_episodes > 0 else 0
    logging.info(f"Validation frames: {len(val_indices)} (avg length: {avg_len_val:.1f})")

    if cfg.seed is not None:
        logging.info(f"Validation split is reproducible with seed={cfg.seed}")
    else:
        logging.info("Validation split is random (no seed provided)")

    return train_indices, val_indices, num_train_episodes, num_val_episodes


def compute_validation_loss(
    policy: PreTrainedPolicy,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Compute validation loss on the validation subset."""
    policy.eval()

    val_loss_sum = 0.0
    val_n_samples = 0
    
    val_start_time = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Forward pass with mixed precision if enabled
            with torch.autocast(device_type=device.type) if use_amp else nullcontext():
                loss, _ = policy.forward(batch)

            val_loss_sum += loss.item() * batch["index"].shape[0]
            val_n_samples += batch["index"].shape[0]

    avg_val_loss = val_loss_sum / val_n_samples if val_n_samples > 0 else 0.0
    val_time = time.perf_counter() - val_start_time
    
    logging.info(f"Validation time: {val_time:.2f}s")
    return {
        "val_loss": avg_val_loss,
        "val_n_samples": val_n_samples,
    }


if __name__ == "__main__":
    init_logging()
    train()
