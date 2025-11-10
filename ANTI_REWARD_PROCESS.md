# Anti-Reward Generation from Trajectory Distribution

## Quick Answer to Your Question

**Yes, you need to train TWO things:**
1. **The anti-reward function** R_aug (always)
2. **The anti-reward policy** π_anti (only for continuous/large MDPs)

---

## The Complete Process

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Expert Policy & Trajectories                           │
│ ───────────────────────────────────────────────────────────────  │
│  Train SAC → Expert trajectories → Occupancy measure x_star    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Train Anti-Reward Function                             │
│ ─────────────────────────────────────────────────────────────── │
│  Train R_aug to maximize distance(x_anti, x_star)              │
│                                                                 │
│  For LINEAR anti-reward:                                       │
│    R_aug(s,a) = w^T φ(s,a)  ← Train weight vector w           │
│                                                                 │
│  Loss: max <x_anti - x_star, R_aug>  (Wasserstein distance)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Get Anti-Reward Policy                                 │
│ ─────────────────────────────────────────────────────────────── │
│  Compute R_new = λ*R + R_aug                                   │
│                                                                 │
│  Discrete MDP:                                                 │
│    π_anti = soft_vi(R_new)  ← NO TRAINING (exact solution)    │
│                                                                 │
│  Continuous MDP:                                               │
│    Train π_anti with SAC using R_new  ← YES TRAINING NEEDED   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Iterate (Optional)                                     │
│ ─────────────────────────────────────────────────────────────── │
│  Collect trajectories from π_anti → x_anti                     │
│  Go back to STEP 2 and retrain R_aug                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Your Specific Question

> "I use SAC RL to get expert policy and expert trajectories, and I want to get linear anti-reward to do MM"

### Answer:

**Training Steps:**

1. **Train R_aug (linear anti-reward function)** ✓ YES
   ```python
   # This is a neural network or linear model
   # Parameters: weight vector w
   # Training: Gradient descent on Wasserstein loss
   ```

2. **Train π_anti (anti-reward policy)** ✓ YES (for continuous)
   ```python
   # This is a NEW SAC policy
   # Parameters: actor & critic networks
   # Training: SAC with augmented reward R_new = λ*R + R_aug
   ```

---

## Code Comparison

### Discrete MDP (Current Codebase)

```python
# policy_randomization.py:Det_WD_baseline

for epoch in range(n_epochs):
    # Train anti-reward network
    for i in range(n_inner_iter):
        loss = -torch.sum((x_tensor - x_star_tensor) * R_aug)
        loss.backward()
        optimizer.step()  # ← TRAIN R_aug

    # Get policy WITHOUT training (exact solution)
    Q, V, policy = soft_vi(env=env, reward=R_aug)  # ← NO TRAINING
```

### Continuous MDP (What You Need)

```python
# Your case with SAC

for epoch in range(n_epochs):
    # Train anti-reward network (same as discrete)
    for i in range(n_inner_iter):
        loss = -torch.sum((x_anti - x_star) * R_aug)
        loss.backward()
        r_aug_optimizer.step()  # ← TRAIN R_aug

    # Train policy WITH RL (different from discrete!)
    anti_policy = SAC("MlpPolicy", env_with_R_aug)
    anti_policy.learn(timesteps=10000)  # ← TRAIN POLICY

    # Update occupancy for next iteration
    x_anti = collect_and_compute_occupancy(anti_policy)
```

---

## Why Two Training Processes?

| Component | What | Why Training Needed |
|-----------|------|---------------------|
| **R_aug** | Anti-reward function | Learn weights to maximize WD from expert |
| **π_anti** (discrete) | Policy | NOT needed - solve MDP exactly with DP |
| **π_anti** (continuous) | Policy | YES needed - cannot solve exactly, must use RL |

---

## Maximum Margin (MM) Context

In Maximum Margin IRL:
- **Margin**: Distance between expert and anti-reward policy
- **Constraint**: Anti-policy achieves E ≥ E_min (e.g., 80% of expert)

The anti-reward R_aug is trained to:
```
max  distance(π_expert, π_anti)
s.t. E_π_anti[R] ≥ E_min
```

This is solved by:
1. Training R_aug with constraint via Lagrangian multiplier λ
2. R_new = λ*R + R_aug balances reward vs. anti-reward

---

## Practical Implementation

See `example_sac_anti_reward.py` for complete implementation.

Key functions:
- `train_expert_sac()` - Train expert with SAC
- `collect_expert_trajectories()` - Get expert demos
- `compute_occupancy_from_trajectories()` - Convert to x_star
- `train_linear_anti_reward()` - Train R_aug weights
- `train_anti_policy()` - **Train π_anti with SAC** (NEW POLICY!)

---

## Summary

✅ **What gets trained:**
1. Anti-reward function R_aug (always)
2. Anti-reward policy π_anti (for continuous MDPs only)

✅ **Your workflow:**
```
SAC expert → trajectories → x_star →
  train R_aug (linear) →
    train π_anti (SAC with R_new) →
      maximum margin achieved!
```
