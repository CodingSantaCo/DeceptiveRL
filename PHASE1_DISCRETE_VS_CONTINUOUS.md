# Phase 1: Discrete vs Continuous MDPs

## The Key Question

> "If we don't train a new policy, how can we update occupancy of anti-reward?"

## Answer: It Depends on MDP Type!

---

## Discrete/Tabular MDPs (Current Codebase)

### Properties:
- **Finite** state and action spaces
- **Known** dynamics: Transition matrix P(s'|s,a)
- **Exact** solutions possible via Dynamic Programming

### Phase 1 Algorithm (Det_WD_baseline)

```python
# policy_randomization.py:220-274

reward_net = WDNNReward(obs_dim)  # Initialize anti-reward network
optimizer = torch.optim.Adam(reward_net.parameters())

for outer_iter in range(n_iter):  # Line 247
    # =====================================================
    # STEP 1: Get policy from current R_aug (NO TRAINING!)
    # =====================================================
    Q, V, policy = soft_vi(env=env, reward=R_aug)  # Line 252
    # What is soft_vi?
    #   - Dynamic Programming algorithm
    #   - Computes optimal soft policy exactly
    #   - Uses Bellman equations: V(s) = logsumexp_a[Q(s,a)]
    #   - NO gradient descent, NO sampling
    #   - Returns: policy π(a|s) for ALL (s,a) pairs

    # ======================================================
    # STEP 2: Get occupancy analytically (NO TRAJECTORIES!)
    # ======================================================
    _, x_1d, x = get_occupancy_measures(env=env, pi=policy)  # Line 253
    # What is get_occupancy_measures?
    #   - Forward propagation through MDP dynamics
    #   - Uses: D(s,a,t) = Σ_{s'} D(s',t-1) * π(a|s') * P(s|s',a)
    #   - Computes EXACT state-action visitation frequencies
    #   - NO trajectory sampling needed!

    # ======================================================
    # STEP 3: Train anti-reward to maximize distance
    # ======================================================
    x_tensor = torch.tensor(x)
    for inner_iter in range(n_iter2):  # Line 260
        optimizer.zero_grad()
        R_aug = reward_net(observation_matrix)

        # Maximize Wasserstein distance: <x - x_star, R_aug>
        loss = -torch.sum((x_tensor - x_star_tensor) * R_aug)  # Line 268
        loss.backward()
        optimizer.step()  # ← TRAIN R_aug network
```

### Key Points:
- **Policy update**: Via exact DP (soft_vi), not training
- **Occupancy update**: Via analytical computation, not trajectories
- **What gets trained**: Only the reward network R_aug

---

## Continuous MDPs (Your SAC Case)

### Properties:
- **Continuous** state/action spaces (or very large discrete)
- **Unknown** or complex dynamics
- **Approximate** solutions only, via RL

### Phase 1 Algorithm (Modified for Continuous)

```python
# Your continuous case with SAC

reward_net = LinearAntiReward(feature_dim)  # Initialize anti-reward
r_optimizer = torch.optim.Adam(reward_net.parameters())

# Initialize policy network (SAC)
current_policy = SAC("MlpPolicy", env)

for outer_iter in range(n_outer_iter):
    # ========================================================
    # STEP 1: Train policy with current R_aug (YES TRAINING!)
    # ========================================================
    # Create environment that uses R_aug as reward
    env_with_R_aug = RewardWrapper(env, reward_fn=compute_R_aug)

    # Train policy to maximize R_aug
    current_policy.set_env(env_with_R_aug)
    current_policy.learn(total_timesteps=5000)  # ← TRAIN POLICY

    # Why training is needed:
    #   - Cannot solve continuous MDP exactly
    #   - Must use RL (SAC) to approximate optimal policy
    #   - Trains actor/critic networks via gradient descent

    # ========================================================
    # STEP 2: Collect trajectories (YES TRAJECTORIES!)
    # ========================================================
    trajectories = collect_trajectories(current_policy, n_episodes=100)
    # Returns: [(s_0, a_0, r_0, ...), (s_1, a_1, r_1, ...), ...]

    # ========================================================
    # STEP 3: Compute occupancy empirically (FROM SAMPLES!)
    # ========================================================
    x_current = compute_occupancy_from_trajectories(trajectories, gamma=0.99)
    # What does this do?
    #   - For each (s,a) in trajectories: accumulate discounted count
    #   - Approximate the true occupancy measure via sampling
    #   - More trajectories = better approximation

    # ========================================================
    # STEP 4: Train anti-reward to maximize distance
    # ========================================================
    x_current_tensor = torch.FloatTensor(x_current)
    for inner_iter in range(n_inner_iter):
        r_optimizer.zero_grad()
        R_aug_values = reward_net(features)

        # Maximize Wasserstein distance
        loss = -torch.sum((x_current_tensor - x_star_tensor) * R_aug_values)
        loss.backward()
        r_optimizer.step()  # ← TRAIN R_aug
```

### Key Points:
- **Policy update**: Via RL training (SAC), cannot use exact DP
- **Occupancy update**: Via trajectory sampling, cannot compute analytically
- **What gets trained**: Both reward network AND policy network

---

## Side-by-Side Comparison

| Aspect | Discrete MDP | Continuous MDP |
|--------|--------------|----------------|
| **State/Action Space** | Finite | Continuous/Large |
| **Dynamics** | Known (P matrix) | Unknown/Complex |
| **Policy Update** | Exact (soft_vi DP) | Approximate (SAC RL) |
| **Policy Training?** | ❌ No | ✅ Yes |
| **Occupancy Computation** | Analytical (forward prop) | Empirical (sampling) |
| **Need Trajectories?** | ❌ No | ✅ Yes |
| **What Gets Trained** | R_aug only | R_aug + policy |
| **Computational Cost** | Low (DP is fast) | High (RL training) |

---

## Code Evidence: How soft_vi Works

From `utils.py:146-219`:

```python
def soft_vi(env, reward, discount, lambda_):
    """Soft Value Iteration - Dynamic Programming"""

    # Initialize V and Q tables
    V = np.full((horizon, n_states), -np.inf)
    Q = np.zeros((horizon, n_states, n_actions))

    # Base case: final timestep
    Q[horizon-1, :, :] = reward  # Just the reward
    V[horizon-1, :] = scipy.special.logsumexp(Q[horizon-1, :, :], axis=1)

    # Backward induction (Dynamic Programming)
    for t in reversed(range(horizon - 1)):
        # Bellman backup using transition matrix
        next_values_s_a = T @ V[t+1, :]  # Matrix multiplication!
        Q[t, :, :] = reward + discount * next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    # Extract soft policy
    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi
```

**Key observation**: This is pure matrix operations, no training loops!

---

## Code Evidence: How get_occupancy_measures Works

From `utils.py:324-374`:

```python
def get_occupancy_measures(env, pi, discount):
    """Compute state-action occupancy analytically"""

    D = np.zeros((horizon + 1, n_states))
    D2 = np.zeros((horizon, n_states, n_actions))
    D[0, :] = env.initial_state_dist  # Starting distribution

    # Forward propagation through MDP dynamics
    for t in range(horizon):
        for a in range(n_actions):
            # D2[t,s,a] = probability of being in (s,a) at time t
            D2[t, :, a] = D[t] * pi[t, :, a]

            # Propagate to next state using transition matrix
            D[t+1, :] += D2[t, :, a] @ T[:, a, :]  # Matrix multiplication!

    # Discount and sum over time
    Dcum2 = discounted_sum(D2, discount)

    return Dcum2  # This is x, the occupancy measure
```

**Key observation**: Uses transition matrix T, no trajectory sampling!

---

## Why You're Confused

Your intuition is **100% correct for continuous MDPs**:

> "We need occupancy of anti-reward, and occupancy is obtained from features from trajectories, right?"

**Yes** - for continuous MDPs!

But the codebase implements **discrete MDPs** where:
- Occupancy is obtained from **exact policy + transition matrix**
- No trajectories needed
- Policy is obtained from **exact DP solution**
- No training needed

---

## For Your SAC Workflow

Since you're using SAC (continuous), you **must** train a policy in Phase 1:

```python
# Phase 1 for continuous MDP (correct approach)

# Outer loop
for epoch in range(n_epochs):
    # 1. Train policy with current R_aug
    policy = train_sac(env, reward_fn=lambda s,a: compute_R_aug(s,a))

    # 2. Collect trajectories
    trajectories = rollout(policy, env, n_episodes=100)

    # 3. Compute occupancy from trajectories
    x_current = compute_occupancy(trajectories)

    # 4. Update R_aug
    # Inner loop
    for inner in range(n_inner):
        loss = -<x_current - x_star, R_aug>
        update(R_aug)
```

This is different from the codebase because **continuous ≠ discrete**!

---

## Summary

**Q**: "If we don't train a new policy, how can we update occupancy?"

**A**:
- **Discrete MDP**: Get new policy via exact DP (soft_vi), compute occupancy analytically → No training needed!
- **Continuous MDP**: Must train new policy via RL (SAC), compute occupancy from trajectories → Training needed!

The codebase is designed for **discrete** MDPs, so it doesn't train policies. For your **continuous** case with SAC, you must add policy training in Phase 1.
