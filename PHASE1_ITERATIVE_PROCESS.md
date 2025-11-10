# Phase 1: The Iterative Update Process

## Your Question

> "In every iteration:
> 1. Update anti-reward by maximizing distance between expert occupancy and anti-occupancy
> 2. Update anti-occupancy using new anti-reward
> 3. Repeat
>
> Is this right? How do I get new occupancy from new reward?"

## Answer: YES! Here's the Exact Process

---

## The Algorithm (Phase 1)

```
Initialize: R_aug (random anti-reward network)

For outer_iter = 1 to n_iter:

    ┌────────────────────────────────────────────────────────┐
    │ STEP 1: Get Policy that MAXIMIZES Current R_aug       │
    └────────────────────────────────────────────────────────┘
    π_anti = argmax_π E_π[R_aug]

    How?
    - Discrete:   π_anti = soft_vi(R_aug)           [Exact DP]
    - Continuous: π_anti = train_SAC(env, R_aug)    [RL training]

    ┌────────────────────────────────────────────────────────┐
    │ STEP 2: Get Occupancy of this Policy                  │
    └────────────────────────────────────────────────────────┘
    x_anti = occupancy(π_anti)

    How?
    - Discrete:   x_anti = get_occupancy_measures(π_anti, P)  [Analytical]
    - Continuous: x_anti = collect_trajectories(π_anti)       [Sampling]

    ┌────────────────────────────────────────────────────────┐
    │ STEP 3: Update R_aug to Maximize Distance             │
    └────────────────────────────────────────────────────────┘
    For inner_iter = 1 to n_iter2:
        loss = -<x_anti - x_expert, R_aug>  // Maximize Wasserstein distance
        R_aug ← R_aug - ∇loss               // Gradient ascent on R_aug

    ┌────────────────────────────────────────────────────────┐
    │ RESULT: R_aug is updated                               │
    └────────────────────────────────────────────────────────┘
    Go to next outer iteration with NEW R_aug
```

---

## Key Insight: What is "Anti-Occupancy"?

**Important**: x_anti is the occupancy of the policy that **MAXIMIZES** R_aug, NOT minimizes!

```
x_anti = occupancy(argmax_π E_π[R_aug])
```

Why? Because we want R_aug such that:
- The policy **maximizing** R_aug...
- ...has occupancy **far from** expert occupancy x_expert

This creates a reward function that "pulls" policies away from expert behavior.

---

## Concrete Example: How to Get New Occupancy from New Reward

### Discrete MDP (Current Codebase)

From `policy_randomization.py:Det_WD_baseline:247-270`:

```python
# Outer loop
for i in range(n_iter+1):  # Line 247

    # ================================================
    # STEP 1: Get policy that MAXIMIZES current R_aug
    # ================================================
    R_aug_np = R_aug.detach().cpu().numpy()  # Current anti-reward

    Q, V, policy = soft_vi(
        env=env,
        lambda_=250,      # Large value → near-deterministic
        discount=gamma,
        reward=R_aug_np   # Use R_aug as the reward!
    )  # Line 252

    # soft_vi solves: π*(s) = argmax_π E_π[R_aug]
    # Returns: policy that MAXIMIZES R_aug

    # ================================================
    # STEP 2: Get occupancy of this policy
    # ================================================
    _, x_1d, x = get_occupancy_measures(
        env=env,
        pi=policy,        # Policy from step 1
        discount=gamma
    )  # Line 253

    # x is now the "anti-occupancy": occupancy of policy maximizing R_aug

    # ================================================
    # STEP 3: Update R_aug to maximize distance
    # ================================================
    x_tensor = torch.tensor(x, device=device, dtype=torch.float32)

    for j in range(n_iter2):  # Inner loop, line 260
        optimizer.zero_grad()

        # Compute R_aug from network
        R_aug = reward_net(observation_matrix)
        R_aug = scale * F.normalize(R_aug.T).T
        R_aug = R_aug.reshape((s_size, a_size))

        # Maximize Wasserstein distance
        # max <x_anti - x_expert, R_aug>
        loss = -torch.sum((x_tensor - x_star_tensor) * R_aug)  # Line 268

        loss.backward()
        optimizer.step()  # Update R_aug network parameters

    # Now R_aug has been updated!
    # Next outer iteration will use NEW R_aug to get NEW policy and NEW occupancy
```

### Continuous MDP (Your SAC Case)

```python
# Initialize
reward_net = LinearAntiReward(feature_dim)
r_optimizer = torch.optim.Adam(reward_net.parameters())

# Outer loop
for outer_iter in range(n_outer_iter):

    # ================================================
    # STEP 1: Train policy that MAXIMIZES current R_aug
    # ================================================
    # Create environment that uses R_aug as reward
    def compute_R_aug(obs, act):
        features = featurize(obs, act)
        with torch.no_grad():
            return reward_net(torch.FloatTensor(features)).item()

    env_with_R_aug = RewardWrapper(
        env,
        reward_fn=compute_R_aug  # Replace env reward with R_aug
    )

    # Train policy to MAXIMIZE R_aug
    anti_policy = SAC("MlpPolicy", env_with_R_aug, verbose=0)
    anti_policy.learn(total_timesteps=5000)

    # anti_policy now approximately maximizes R_aug

    # ================================================
    # STEP 2: Get occupancy by collecting trajectories
    # ================================================
    trajectories = collect_trajectories(
        anti_policy,
        env_name,
        n_episodes=100
    )

    # Compute empirical occupancy from trajectories
    x_anti, weights = compute_occupancy_from_trajectories(
        trajectories,
        gamma=0.99
    )

    # x_anti is the "anti-occupancy": occupancy of policy maximizing R_aug

    # ================================================
    # STEP 3: Update R_aug to maximize distance
    # ================================================
    # Create features for all sampled (s,a) pairs
    sa_pairs = np.concatenate([
        np.concatenate([traj.obs, traj.acts], axis=-1)
        for traj in trajectories
    ])
    features = featurize(sa_pairs)

    features_tensor = torch.FloatTensor(features)
    weights_tensor = torch.FloatTensor(weights)

    # Also need expert features (computed once at start)
    expert_features_tensor = torch.FloatTensor(expert_features)
    expert_weights_tensor = torch.FloatTensor(expert_weights)

    for inner_iter in range(n_inner_iter):
        r_optimizer.zero_grad()

        # Compute R_aug values
        R_aug_anti = reward_net(features_tensor)      # R_aug on anti-policy states
        R_aug_expert = reward_net(expert_features_tensor)  # R_aug on expert states

        # Maximize Wasserstein distance:
        # max E_{x_anti}[R_aug] - E_{x_expert}[R_aug]
        # = max <x_anti - x_expert, R_aug>
        loss = -(
            (R_aug_anti * weights_tensor).sum() -
            (R_aug_expert * expert_weights_tensor).sum()
        )

        loss.backward()
        r_optimizer.step()  # Update R_aug network parameters

    # Now R_aug has been updated!
    # Next outer iteration will:
    #   - Train NEW policy with NEW R_aug
    #   - Get NEW occupancy from NEW policy
    #   - Update R_aug again
```

---

## Step-by-Step Flow Diagram

```
Iteration 1:
    R_aug⁽¹⁾ (initial)
        ↓
    Train/Solve: π_anti⁽¹⁾ = argmax E[R_aug⁽¹⁾]
        ↓
    Compute: x_anti⁽¹⁾ = occupancy(π_anti⁽¹⁾)
        ↓
    Update: R_aug⁽²⁾ ← maximize <x_anti⁽¹⁾ - x_expert, R_aug>

Iteration 2:
    R_aug⁽²⁾ (updated from iteration 1)
        ↓
    Train/Solve: π_anti⁽²⁾ = argmax E[R_aug⁽²⁾]  ← NEW policy!
        ↓
    Compute: x_anti⁽²⁾ = occupancy(π_anti⁽²⁾)    ← NEW occupancy!
        ↓
    Update: R_aug⁽³⁾ ← maximize <x_anti⁽²⁾ - x_expert, R_aug>

Iteration 3:
    R_aug⁽³⁾ (updated from iteration 2)
        ↓
    Train/Solve: π_anti⁽³⁾ = argmax E[R_aug⁽³⁾]  ← NEW policy!
        ↓
    Compute: x_anti⁽³⁾ = occupancy(π_anti⁽³⁾)    ← NEW occupancy!
        ↓
    ...

Final:
    R_aug⁽ᶠⁱⁿᵃˡ⁾ such that:
        π* = argmax E[R_aug⁽ᶠⁱⁿᵃˡ⁾]
        has occupancy x* far from x_expert
```

---

## Why This Works: The Wasserstein Distance Perspective

The Wasserstein distance between two distributions has a dual formulation:

```
W(x_anti, x_expert) = max_{f: Lipschitz} E_{x_anti}[f] - E_{x_expert}[f]
                     = max_f <x_anti - x_expert, f>
```

In our case:
- `f` is the anti-reward function `R_aug`
- We're finding `R_aug` that maximizes this distance
- But `x_anti` depends on `R_aug` (it's the occupancy of policy maximizing `R_aug`)

So we alternate:
1. Fix `R_aug`, find policy maximizing it, get `x_anti` (outer loop)
2. Fix `x_anti`, update `R_aug` to maximize distance (inner loop)
3. Repeat until convergence

This is a **minimax optimization**:

```
max_{R_aug} min_π <occupancy(π) - x_expert, R_aug>
```

where π maximizes E[R_aug].

---

## Summary

**Q1**: "Is this process right?"

**A**: ✅ YES! The process is:
```
Loop:
  1. Get policy that MAXIMIZES current R_aug
  2. Get occupancy of this policy (this is x_anti)
  3. Update R_aug to maximize <x_anti - x_expert, R_aug>
  4. Repeat with updated R_aug
```

**Q2**: "How can I get new occupancy from new reward?"

**A**:
- **Get policy**: Find π that maximizes R_aug (via soft_vi or SAC training)
- **Get occupancy**:
  - Discrete: Analytical computation from π and transition matrix
  - Continuous: Sample trajectories from π and compute empirical occupancy

The key is: **R_aug changes → policy changes → occupancy changes**

This creates an iterative process that finds an anti-reward whose optimal policy is far from the expert.
