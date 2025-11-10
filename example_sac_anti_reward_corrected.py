"""
CORRECTED: Generate Linear Anti-Reward from SAC Expert Policy

Two Separate Phases:
1. Generate anti-reward R_aug (train to maximize distance)
2. Apply MM algorithm (find λ using fixed R_aug)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from imitation.data import rollout
from typing import Tuple

# ============================================================================
# PHASE 1: GENERATE ANTI-REWARD (Maximize Distance)
# ============================================================================

def collect_trajectories(policy, env_name: str, n_episodes: int = 50):
    """Collect trajectories from a policy."""
    venv = gym.make_vec(env_name, num_envs=1)
    demos = rollout.rollout(
        policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_episodes),
        rng=np.random.default_rng(42),
    )
    return demos


def compute_occupancy_from_trajectories(demos, gamma: float = 0.99):
    """Compute discounted occupancy measure from trajectories."""
    # Simplified: just collect (s, a) pairs with discount weights
    sa_pairs = []
    weights = []

    for traj in demos:
        discount = 1.0
        for obs, act in zip(traj.obs, traj.acts):
            sa_pairs.append(np.concatenate([obs, act]))
            weights.append(discount)
            discount *= gamma

    sa_pairs = np.array(sa_pairs)
    weights = np.array(weights)
    weights /= weights.sum()  # Normalize

    return sa_pairs, weights


class LinearAntiReward(nn.Module):
    """Linear anti-reward: R_aug(s,a) = w^T φ(s,a)"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze()


def featurize(sa_pairs: np.ndarray, n_rbf: int = 20) -> np.ndarray:
    """Create RBF features."""
    dim = sa_pairs.shape[-1]
    np.random.seed(42)
    centers = np.random.randn(n_rbf, dim) * 0.5

    features = []
    for center in centers:
        dist_sq = np.sum((sa_pairs - center) ** 2, axis=-1, keepdims=True)
        features.append(np.exp(-dist_sq / (2 * 0.5 ** 2)))

    return np.concatenate(features, axis=-1)


def phase1_generate_anti_reward(
    expert_policy: SAC,
    env_name: str,
    n_outer_iter: int = 20,
    n_inner_iter: int = 100,
    lr: float = 1e-3
) -> Tuple[LinearAntiReward, callable]:
    """
    PHASE 1: Generate anti-reward by maximizing Wasserstein distance.

    This corresponds to Det_WD_baseline in the codebase.
    """
    print("=" * 70)
    print("PHASE 1: GENERATE ANTI-REWARD (Maximize Distance from Expert)")
    print("=" * 70)

    # Step 1: Get expert occupancy
    print("\n[1/3] Collecting expert trajectories...")
    expert_demos = collect_trajectories(expert_policy, env_name, n_episodes=100)
    expert_sa, expert_weights = compute_occupancy_from_trajectories(expert_demos)
    print(f"  ✓ Collected {len(expert_sa)} state-action pairs")

    # Step 2: Create features
    print("\n[2/3] Creating features...")
    expert_features = featurize(expert_sa, n_rbf=20)
    feature_dim = expert_features.shape[-1]
    print(f"  ✓ Feature dimension: {feature_dim}")

    # Compute weighted expert occupancy in feature space
    x_star = (expert_features.T @ expert_weights).astype(np.float32)
    x_star_tensor = torch.FloatTensor(x_star)

    # Step 3: Train anti-reward
    print(f"\n[3/3] Training anti-reward network ({n_outer_iter} outer iterations)...")

    anti_reward_net = LinearAntiReward(feature_dim)
    optimizer = torch.optim.Adam(anti_reward_net.parameters(), lr=lr)

    # Create environment for training current policy
    env = gym.make(env_name)

    # Initialize current policy (could be random or copy of expert)
    current_policy = SAC("MlpPolicy", env, verbose=0)

    for outer_iter in range(n_outer_iter):
        # Step A: Get current policy's occupancy
        # For discrete MDP: current_policy = soft_vi(R_aug)
        # For continuous MDP: need to train current_policy
        if outer_iter > 0:  # Skip first iteration
            # Train policy with current R_aug for a few steps
            # This is the key difference for continuous MDPs!
            with torch.no_grad():
                # Create reward function wrapper
                class AntiRewardWrapper(gym.Wrapper):
                    def __init__(self, env, anti_reward_net, features_fn):
                        super().__init__(env)
                        self.anti_reward_net = anti_reward_net
                        self.features_fn = features_fn
                        self.last_obs = None

                    def reset(self, **kwargs):
                        obs, info = self.env.reset(**kwargs)
                        self.last_obs = obs
                        return obs, info

                    def step(self, action):
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        # Use R_aug as reward
                        sa = np.concatenate([self.last_obs, action])
                        feat = self.features_fn(sa.reshape(1, -1))
                        r_aug = self.anti_reward_net(torch.FloatTensor(feat)).item()
                        self.last_obs = obs
                        return obs, r_aug, terminated, truncated, info

                wrapped_env = AntiRewardWrapper(
                    gym.make(env_name),
                    anti_reward_net,
                    lambda sa: featurize(sa, n_rbf=20)
                )
                current_policy.set_env(wrapped_env)
                current_policy.learn(total_timesteps=2000, reset_num_timesteps=False)

        # Collect trajectories from current policy
        current_demos = collect_trajectories(current_policy, env_name, n_episodes=50)
        current_sa, current_weights = compute_occupancy_from_trajectories(current_demos)
        current_features = featurize(current_sa, n_rbf=20)

        # Compute weighted occupancy
        x_current = (current_features.T @ current_weights).astype(np.float32)
        x_current_tensor = torch.FloatTensor(x_current)

        # Step B: Train anti-reward to maximize distance
        for inner_iter in range(n_inner_iter):
            optimizer.zero_grad()

            # Compute anti-reward on expert features (for evaluation)
            r_aug_expert = anti_reward_net(torch.FloatTensor(expert_features))
            r_aug_current = anti_reward_net(torch.FloatTensor(current_features))

            # Wasserstein distance loss: max <x_current - x_star, R_aug>
            # Equivalent to: E_current[R_aug] - E_expert[R_aug]
            loss = -(
                (r_aug_current * torch.FloatTensor(current_weights)).sum() -
                (r_aug_expert * torch.FloatTensor(expert_weights)).sum()
            )

            loss.backward()
            optimizer.step()

        if outer_iter % 5 == 0:
            print(f"  Outer iter {outer_iter}: WD = {-loss.item():.4f}")

    print("\n  ✓ Anti-reward network trained!")
    print(f"  Final Wasserstein distance: {-loss.item():.4f}\n")

    # Create feature extractor function
    def compute_anti_reward(obs, act):
        sa = np.concatenate([obs, act], axis=-1)
        features = featurize(sa, n_rbf=20)
        with torch.no_grad():
            return anti_reward_net(torch.FloatTensor(features)).numpy()

    return anti_reward_net, compute_anti_reward


# ============================================================================
# PHASE 2: MAXIMUM MARGIN ALGORITHM (Find λ)
# ============================================================================

class AugmentedRewardWrapper(gym.Wrapper):
    """Wrapper: R_new = λ*R + R_aug"""

    def __init__(self, env, compute_anti_reward, lambda_weight: float):
        super().__init__(env)
        self.compute_anti_reward = compute_anti_reward
        self.lambda_weight = lambda_weight
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # R_new = λ*R + R_aug
        r_aug = self.compute_anti_reward(
            self.last_obs.reshape(1, -1),
            action.reshape(1, -1)
        )
        augmented_reward = self.lambda_weight * reward + r_aug

        self.last_obs = obs
        info['original_reward'] = reward
        info['anti_reward'] = r_aug
        info['augmented_reward'] = augmented_reward

        return obs, augmented_reward, terminated, truncated, info


def evaluate_true_reward(policy, env_name: str, n_episodes: int = 10):
    """Evaluate policy on TRUE reward (not augmented)."""
    env = gym.make(env_name)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward  # True reward

        returns.append(episode_return)

    return np.mean(returns)


def phase2_maximum_margin(
    compute_anti_reward: callable,
    env_name: str,
    E_min: float,
    n_iter: int = 10,
    train_steps_per_iter: int = 10000
) -> Tuple[SAC, float]:
    """
    PHASE 2: Maximum Margin algorithm to find optimal λ.

    R_aug is FIXED (from Phase 1), only λ is adjusted.

    This corresponds to Det_WD in the codebase (lines 169-214).
    """
    print("=" * 70)
    print("PHASE 2: MAXIMUM MARGIN ALGORITHM (Find λ with Fixed R_aug)")
    print("=" * 70)
    print(f"\nConstraint: E[R_true] >= {E_min:.2f}\n")

    # Binary search for λ
    lambda_low = 0.0
    lambda_high = 10.0

    best_policy = None
    best_lambda = None

    for iteration in range(n_iter):
        lambda_mid = (lambda_low + lambda_high) / 2

        print(f"[Iteration {iteration+1}/{n_iter}] Testing λ = {lambda_mid:.3f}")

        # Create environment with augmented reward
        env = gym.make(env_name)
        env_aug = AugmentedRewardWrapper(env, compute_anti_reward, lambda_mid)

        # Train policy with R_new = λ*R + R_aug
        # R_aug is FIXED, only λ changes!
        policy = SAC("MlpPolicy", env_aug, verbose=0)
        policy.learn(total_timesteps=train_steps_per_iter)

        # Evaluate on TRUE reward R (not augmented!)
        E_true = evaluate_true_reward(policy, env_name, n_episodes=10)

        print(f"  E[R_true] = {E_true:.2f}, Target = {E_min:.2f}")

        # Adjust λ using binary search
        if E_true > E_min + 0.1:  # Small tolerance
            # Reward too high, decrease λ (give more weight to anti-reward)
            lambda_high = lambda_mid
            print(f"  → Too high, decreasing λ (new range: [{lambda_low:.3f}, {lambda_mid:.3f}])")
        elif E_true < E_min - 0.1:
            # Reward too low, increase λ (give more weight to true reward)
            lambda_low = lambda_mid
            print(f"  → Too low, increasing λ (new range: [{lambda_mid:.3f}, {lambda_high:.3f}])")
        else:
            # Within tolerance
            print(f"  ✓ Constraint satisfied!")
            best_policy = policy
            best_lambda = lambda_mid
            break

        if iteration == n_iter - 1:
            best_policy = policy
            best_lambda = lambda_mid

        print()

    print(f"\n  ✓ Final λ = {best_lambda:.3f}")
    print(f"  ✓ Final E[R_true] = {E_true:.2f}\n")

    return best_policy, best_lambda


# ============================================================================
# MAIN: Complete Workflow
# ============================================================================

def main():
    """
    Complete workflow:
    1. Train SAC expert
    2. PHASE 1: Generate linear anti-reward (train to maximize distance)
    3. PHASE 2: Apply MM algorithm (find λ with fixed R_aug)
    """

    env_name = "Pendulum-v1"

    print("=" * 70)
    print("STEP 0: TRAIN EXPERT POLICY")
    print("=" * 70)
    env = gym.make(env_name)
    expert_policy = SAC("MlpPolicy", env, verbose=1)
    expert_policy.learn(total_timesteps=20000)
    E_expert = evaluate_true_reward(expert_policy, env_name, n_episodes=20)
    print(f"\n✓ Expert trained! E[R_expert] = {E_expert:.2f}\n")

    # PHASE 1: Generate anti-reward
    anti_reward_net, compute_anti_reward = phase1_generate_anti_reward(
        expert_policy,
        env_name,
        n_outer_iter=10,
        n_inner_iter=50,
        lr=1e-3
    )

    # PHASE 2: Maximum Margin algorithm
    E_min = 0.7 * E_expert  # Target: 70% of expert performance
    final_policy, final_lambda = phase2_maximum_margin(
        compute_anti_reward,
        env_name,
        E_min=E_min,
        n_iter=8,
        train_steps_per_iter=5000
    )

    # Final evaluation
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    E_final = evaluate_true_reward(final_policy, env_name, n_episodes=20)
    print(f"Expert return:       {E_expert:.2f}")
    print(f"Anti-reward return:  {E_final:.2f}")
    print(f"Target (E_min):      {E_min:.2f}")
    print(f"Ratio:               {E_final / E_expert:.1%}")
    print(f"Optimal λ:           {final_lambda:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ PHASE 1: Trained linear anti-reward R_aug to maximize distance")
    print("✓ PHASE 2: Found λ such that R_new = λ*R + R_aug satisfies constraint")
    print("✓ Final policy maximizes distance while maintaining E >= E_min")
    print("=" * 70)


if __name__ == "__main__":
    main()
