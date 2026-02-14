# RL Fine-tuning of IL Policies (ACT/SmolVLA)

## Context

We have a working **SAC (Soft Actor-Critic)** implementation for real robot RL with human-in-the-loop interventions (HILSerl). SAC trains from scratch using a Gaussian stochastic policy.

We want to **pretrain with imitation learning** (using ACT or SmolVLA) and then **fine-tune with RL** to improve beyond the demonstration data.

**Key policies:**
- **ACT (Action Chunking Transformer):** Predicts chunks of 50-100 actions. Uses VAE internally.
- **SmolVLA:** Vision-language-action model using flow matching. Predicts action chunks.

**Our setup:**
- Real robot manipulation (SO-100 arm)
- Human-in-the-loop: humans can intervene and provide rewards via keyboard/gamepad
- Existing SAC infrastructure with critic ensemble, target networks, replay buffer

---

## Solution: Residual RL (ResFiT)

Based on [Ankile et al. 2025 - "Residual Off-Policy RL for Finetuning BC Policies"](https://arxiv.org/html/2509.19301v1)

**Key insight:** Don't backprop through ACT/SmolVLA. Freeze it, learn residual corrections with SAC.

```
final_action = BC_policy(obs) + clip(residual_policy(obs), -δ, δ)
                   ↑                        ↑
            ACT/SmolVLA              Small MLP (SAC)
             (FROZEN)                 (trainable)
```

### Why This Works
- BC policy is black-box → works with ANY architecture
- Residual is small MLP → sample efficient, fast to train
- Bounded residuals → safe exploration, stays close to BC behavior
- No forgetting → BC weights frozen

### Technical Details from Paper

**Residual bounds (δ):**
- Task-specific tuning required
- Paper uses values that allow ~10-30% correction of action range
- Start conservative, increase if policy plateaus

**Architecture:**
- Residual policy: small MLP (256 hidden units, 2-3 layers)
- Critic: standard SAC critic, sees (s, a_total) not (s, a_base, a_residual)
- Off-policy: REDQ-style (randomized ensemble) for sample efficiency

**Key tricks from paper:**
1. **Per-step residuals** even with action-chunked BC (query BC every step, not every chunk)
2. **Demonstrations in replay buffer** - mix BC demos with online data
3. **Sparse binary rewards** - just success/failure, no dense shaping needed
4. **High UTD (update-to-data ratio)** - many gradient steps per env step

**What they achieved:**
- First real-world RL on humanoid with dexterous hands
- Works with diffusion policies, action chunking, large transformers
- Sample efficient enough for real robots

---

## Implementation Plan

### Stage 1: Human Reward System
Get RL working with human-provided rewards via keyboard.

**Reward keys:**
- `1-9` → graded reward (+0.1 to +0.9)
- `0` → zero/neutral (+0.0)

- [ ] Add `HUMAN_REWARD` to `TeleopEvents` (stores reward value, not just bool)
- [ ] Add number key handling (0-9) in keyboard teleop
- [ ] Create `HumanRewardProcessorStep`
- [ ] Make reward classifier optional in pipeline
- [ ] Test: run existing SAC with human-only rewards

### Stage 2: SAC Baseline with Human Rewards
Verify SAC learns something with human feedback before adding complexity.

- [ ] Train SAC from scratch on simple task with human rewards
- [ ] Tune reward magnitude and timing
- [ ] Establish baseline success rate

### Stage 3: Residual RL (ResFiT)
Add pretrained BC policy with residual learning.

- [ ] Create `ResidualRLPolicy` class
- [ ] Load pretrained ACT/SmolVLA as frozen base
- [ ] Add residual actor (small MLP, SAC-style)
- [ ] Implement: `action = base_policy(obs) + clip(residual(obs), -δ, δ)`
- [ ] Critic sees (s, a_total) only
- [ ] Per-step residuals (query BC every step)

### Stage 4: Extensions
- [ ] Add BC demos to replay buffer
- [ ] Tune residual bounds (δ)
- [ ] Compare: human reward vs classifier reward
- [ ] Try with SmolVLA as base policy

---

## Architecture: ResidualRLPolicy

```
ResidualRLPolicy
├── base_policy: ACTPolicy | SmolVLAPolicy (FROZEN)
│   └── predict_action_chunk() → use first action
├── residual_actor: MLP (small, TRAINABLE)
│   └── outputs bounded correction
├── critic_ensemble (TRAINABLE)
│   └── Q(s, a_total)
├── critic_target
└── log_alpha (temperature)

select_action(obs):
    base_action = base_policy.select_action(obs)  # from chunk
    residual = residual_actor(obs)
    residual = clip(residual, -delta, delta)
    return base_action + residual
```

## Key Decisions
- **Base policy:** Frozen ACT or SmolVLA
- **Residual:** Small MLP, bounded output
- **Action selection:** Query BC every step, add residual every step
- **Critic input:** (s, a_total) only - simpler
- **Exploration:** Via residual policy's stochasticity (SAC entropy)
