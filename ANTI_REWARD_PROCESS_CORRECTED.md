# Anti-Reward Generation from Trajectory Distribution (CORRECTED)

## Two Separate Phases

### Phase 1: Generate Anti-Reward
**Goal**: Train R_aug to **maximize distance** from expert distribution

### Phase 2: Maximum Margin Algorithm
**Goal**: Find λ to **satisfy reward constraint** using the fixed R_aug

---

## Detailed Process

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: GENERATE ANTI-REWARD (Det_WD_baseline)                │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  Expert occupancy measure x_star                        │
│ Train:  Reward network R_aug                                   │
│ Goal:   Maximize Wasserstein distance from x_star             │
│                                                                 │
│ Algorithm:                                                      │
│   for epoch in range(n_iter):                                  │
│     1. Compute policy: π = solve_MDP(R_aug)  ← NOT training   │
│     2. Get occupancy:  x = occupancy(π)                        │
│     3. Train R_aug:                                            │
│        for inner_iter in range(n_iter2):                       │
│          loss = -<x - x_star, R_aug>  ← Maximize distance     │
│          update R_aug parameters                               │
│                                                                 │
│ Output: R_aug (fixed anti-reward function)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: MAXIMUM MARGIN ALGORITHM (Det_WD)                     │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  R_aug from Phase 1 (FIXED, no more training)          │
│ Find:   Optimal λ                                              │
│ Goal:   Satisfy constraint E[R] = E_min                        │
│                                                                 │
│ Algorithm:                                                      │
│   for i in range(n_iter):                                      │
│     1. R_new = λ*R + R_aug  ← R_aug is FIXED                  │
│     2. π = solve_MDP(R_new)                                    │
│     3. E = expected_reward(π, R)                               │
│     4. Adjust λ:                                               │
│        if E > E_min: decrease λ                                │
│        if E < E_min: increase λ                                │
│                                                                 │
│ Output: Final policy π and λ satisfying constraint             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Code Locations

### Phase 1: `Det_WD_baseline` (policy_randomization.py:220-274)

```python
def Det_WD_baseline(env, P, R, s_size, a_size, alpha, gamma, x_star, ...):
    """Generate anti-reward by maximizing Wasserstein distance."""

    # Initialize reward network
    reward_net = WDNNReward(obs_dim, layers_data=layers_data)  # Line 234
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr_wd)

    for i in range(n_iter+1):  # Outer loop (Line 247)
        # Get policy for current R_aug (NOT training policy!)
        Q, V, policy = soft_vi(env=env, reward=R_aug)  # Line 252
        _, x_1d, x = get_occupancy_measures(env=env, pi=policy)  # Line 253

        # Train anti-reward network (inner loop)
        for j in range(n_iter2):  # Line 260
            optimizer.zero_grad()
            R_aug = reward_net(observation_matrix)

            # Maximize Wasserstein distance
            loss = -torch.sum((x_tensor - x_star_tensor) * R_aug)  # Line 268
            loss.backward()
            optimizer.step()  # Line 270 ← TRAIN R_aug

    return R_aug, reward_net, exp_baseline
```

**What's trained?** Only the reward network (R_aug parameters)
**Policy?** Computed via soft_vi (dynamic programming, NOT trained)

---

### Phase 2: `Det_WD` (policy_randomization.py:118-217)

```python
def Det_WD(env, P, R, s_size, a_size, alpha, gamma, E_min, x_star, ...):
    """Maximum Margin algorithm to find optimal λ."""

    # PHASE 1: Generate anti-reward ONCE
    _, reward_net, exp_baseline = Det_WD_baseline(...)  # Line 140

    # Extract R_aug (now FIXED)
    R_aug = reward_net(observation_matrix)  # Line 151
    R_aug = R_aug.detach().cpu().numpy()  # Line 153

    # PHASE 2: MM algorithm - find λ
    lambda_ = initialize_lambda()

    for i in range(n_iter+1):  # Line 169
        # Combine rewards (R_aug is FIXED!)
        R_new = lambda_ * R + R_aug  # Line 170

        # Solve MDP (NOT training)
        _, _, policy = soft_vi(env=env, reward=R_new)  # Line 172
        _, x_1d, x = get_occupancy_measures(env=env, pi=policy)
        exp_reward = x.flatten().T @ R.flatten()  # Line 175

        # Check constraint
        grad_lambda = exp_reward - E_min  # Line 179

        # Adjust λ (binary search or gradient descent)
        if binary_search:  # Lines 202-207
            if grad_lambda > 0:
                r = lambda_  # Reward too high, decrease λ
            else:
                l = lambda_  # Reward too low, increase λ
            lambda_ = (l + r) / 2
        else:  # Line 210
            lambda_ -= lr * grad_lambda

    return policy, x, R_new
```

**What's trained?** NOTHING - just searching for λ
**R_aug?** Fixed from Phase 1
**Policy?** Computed via soft_vi (not trained)

---

## For Your SAC Case

### Phase 1: Generate Linear Anti-Reward

```python
# 1. Get expert trajectories
expert_sac = SAC("MlpPolicy", env, ...)
expert_sac.learn(total_timesteps=100000)
expert_demos = collect_trajectories(expert_sac)
x_star = compute_occupancy(expert_demos)

# 2. Train linear anti-reward R_aug(s,a) = w^T φ(s,a)
class LinearAntiReward(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.linear(features)

anti_reward_net = LinearAntiReward(feature_dim)
optimizer = torch.optim.Adam(anti_reward_net.parameters())

for epoch in range(n_epochs):
    # Get current policy (for continuous, need RL here)
    current_policy = train_or_update_policy(env, R_aug)  # ← Need RL
    current_demos = collect_trajectories(current_policy)
    x_current = compute_occupancy(current_demos)

    # Train anti-reward to maximize distance
    for _ in range(n_inner_iter):
        loss = -torch.sum((x_current - x_star) * R_aug_values)
        loss.backward()
        optimizer.step()  # ← TRAIN R_aug

# R_aug is now FIXED
```

### Phase 2: Maximum Margin Algorithm

```python
# Now R_aug is FIXED, find optimal λ

# Binary search for λ
lambda_low = 0.0
lambda_high = 10.0

for iteration in range(n_iter):
    lambda_mid = (lambda_low + lambda_high) / 2

    # Train policy with augmented reward
    env_aug = AugmentedRewardEnv(env, lambda_mid, R_aug)  # R_new = λ*R + R_aug
    policy = SAC("MlpPolicy", env_aug, ...)
    policy.learn(total_timesteps=10000)  # ← TRAIN POLICY (Phase 2 only!)

    # Evaluate on TRUE reward R
    E = evaluate_policy(policy, env, R)

    # Adjust λ
    if E > E_min:
        lambda_high = lambda_mid  # Decrease λ
    else:
        lambda_low = lambda_mid   # Increase λ

# Final policy achieves E ≈ E_min
```

---

## What Gets Trained?

| Phase | Discrete MDP | Continuous MDP (SAC) |
|-------|--------------|----------------------|
| **Phase 1: Generate R_aug** | Train R_aug network | Train R_aug network |
| **Phase 1: Get policy** | soft_vi (exact) | Train with RL |
| **Phase 2: Find λ** | Adjust λ only | Adjust λ only |
| **Phase 2: Get policy** | soft_vi (exact) | Train with RL |

---

## Summary

✅ **Two separate phases:**
1. **Anti-reward generation**: Train R_aug to maximize distance
2. **Maximum Margin**: Use fixed R_aug, find λ to satisfy constraint

✅ **What gets trained in each phase:**

**Phase 1 (Anti-reward generation):**
- Discrete: Train R_aug network (policy via exact solution)
- Continuous: Train R_aug network + policy with RL

**Phase 2 (MM algorithm):**
- Discrete: Just search for λ (policy via exact solution)
- Continuous: Search for λ + train policy with RL

✅ **Key insight:** R_aug is trained ONCE in Phase 1, then FIXED in Phase 2

---

## Code References

- Phase 1 anti-reward: `policy_randomization.py:Det_WD_baseline:220-274`
- Phase 2 MM algorithm: `policy_randomization.py:Det_WD:138-217`
- Line 140: Call to Det_WD_baseline (Phase 1)
- Line 151: Extract fixed R_aug
- Line 169-214: MM loop with fixed R_aug (Phase 2)
