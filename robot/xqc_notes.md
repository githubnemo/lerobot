# XQC: Well-Conditioned Optimization Accelerates Deep RL

## Key Technical Details of XQC

XQC is designed to improve the optimization landscape of the critic network (specifically the condition number of the Hessian) rather than just scaling up parameters. It does this through a synergistic combination of three main components:

### 1. Architecture & Batch Normalization (BN)
*   **Block Design:** `Dense(dim) -> BatchNorm -> ReLU`.
*   **Placement:** BN is applied directly to the network input and after each linear layer (before the activation).
*   **Joined Forward Pass:** To make BN work in off-policy RL, XQC uses a joined forward pass for both current $(s, a)$ and next $(s', a')$ transitions to calculate consistent BN running statistics.
*   **Network Size:** 
    *   Critic: 4 hidden layers, 512 neurons each.
    *   Actor: 4 hidden layers, 256 neurons each.

### 2. Weight Normalization (WN)
*   **Mechanism:** After each gradient step, the weights of each dense layer are projected to the unit sphere.
*   **Exceptions:** The final output layers (e.g., `mu`, `log_std`, `head`) are *not* normalized.
*   **Purpose:** Keeps the effective learning rate (ELR) constant, preventing loss of plasticity.

### 3. Distributional Critic with Cross-Entropy (CE) Loss
*   **Mechanism:** Uses a C51-style categorical critic with 101 atoms instead of scalar MSE regression.
*   **Support:** Bounded to `[-5, 5]`.
*   **Reward Normalization:** Standard reward normalization (using running standard deviation of returns) is used to bound the Q-values to fit within the `[-5, 5]` support.
*   **Purpose:** The CE loss naturally bounds gradient norms, which, combined with WN, keeps the effective learning rate remarkably stable.

### 4. Key Hyperparameters (Proprioception / MuJoCo)
*   **UTD Ratio:** 2
*   **Policy Update Delay:** 3 (Actor is updated every 3 critic updates)
*   **Batch Size:** 256
*   **Learning Rates:** 0.0003 (Actor, Critic, Temperature)
*   **Target Entropy:** $-|A| / 2$
*   **Target Network Tau:** 0.005
*   **Optimizer:** Adam

---

## Target Task: `HalfCheetah-v4` (MuJoCo)

`HalfCheetah-v4` is a standard, continuous control benchmark that is complex enough to test the optimization dynamics but fast to simulate. XQC evaluates on 6 MuJoCo tasks including this one.

### Expected Performance Milestones
While the exact numerical curve for HalfCheetah isn't isolated in the provided text (it's aggregated in the AUC scores), based on standard MuJoCo benchmarks and XQC's sample efficiency claims:
*   **100k steps:** Should show strong early learning (often >3,000 - 5,000 return).
*   **500k steps:** Should be nearing convergence (often >8,000 - 10,000 return).
*   **1M steps (Final):** State-of-the-art performance (typically 12,000 - 15,000+ return on HalfCheetah).
*   **Key Metric to Watch:** The stability of the Effective Learning Rate (ELR) and Gradient Norms. In our script, watch `Lq` (Critic Loss) and `La` (Actor Loss) to see if they remain stable rather than exploding.