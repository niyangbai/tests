import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

class MappingEnv(gym.Env):
    """
    Gym environment for cash-flow mapping. 
    Enforces that next flow's start‐bucket >= previous flow's end‐bucket.
    """
    metadata = {'render.modes': []}

    def __init__(self, array1, array2, durations, bins=11):
        super().__init__()
        self.array1 = np.asarray(array1, dtype=float)
        self.array2 = np.asarray(array2, dtype=float)
        self.durs   = np.asarray(durations, dtype=float) / 10000.0
        self.N = len(self.array1)
        self.M = len(self.durs)
        self.bins = bins

        self.action_space = spaces.Discrete(self.M * self.bins)
        low  = np.array([0.0, -1e6, -1e6] + [0]*self.M, dtype=float)
        high = np.array([1.0,  1e6,  1e6] + [1]*self.M, dtype=float)
        self.observation_space = spaces.Box(low=low, high=high, dtype=float)
        self.reset()

    def reset(self):
        self.j = 0
        self.resid_DV01 = 0.0
        self.resid_PV   = 0.0
        self.last_i = 0     # start of first flow
        return self._get_obs()

    def step(self, action):
        # decode
        raw_i = action // self.bins
        b     = action % self.bins
        x     = b / (self.bins - 1)

        # enforce monotonic start >= prev end
        i = max(raw_i, self.last_i)
        i_next = min(i+1, self.M-1)

        # exposures
        dv01 = self.array1[self.j]
        pv   = self.array2[self.j]
        hedged_dv01 = x*(dv01 + pv*self.durs[i]) + (1-x)*(dv01 + pv*self.durs[i_next])
        hedged_pv   = x*pv + (1-x)*pv

        self.resid_DV01 += (dv01 - hedged_dv01)
        self.resid_PV   += (pv   - hedged_pv)

        # update for next
        self.last_i = i_next   # <--- previous end becomes new start
        self.j += 1
        done = (self.j >= self.N)

        if done:
            mismatch = np.sqrt(self.resid_DV01**2 + self.resid_PV**2)
            reward = -mismatch
        else:
            reward = 0.0

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            [self.j/self.N, self.resid_DV01, self.resid_PV],
            np.ones(self.M)
        ]).astype(float)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        return self.policy_head(x)


def train_reinforce(env, policy, optimizer, episodes=2000, gamma=0.99):
    for ep in range(episodes):
        obs = env.reset()
        log_probs, rewards = [], []
        prev_mismatch = None
        done = False

        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))

            obs, _, done, _ = env.step(a.item())
            # shape reward by reduction in mismatch
            current_mismatch = np.sqrt(env.resid_DV01**2 + env.resid_PV**2)
            if prev_mismatch is None:
                rewards.append(0.0)
            else:
                rewards.append(prev_mismatch - current_mismatch)
            prev_mismatch = current_mismatch

        # terminal reduction
        final_mismatch = np.sqrt(env.resid_DV01**2 + env.resid_PV**2)
        rewards[-1] += prev_mismatch - final_mismatch

        # compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = returns - returns.mean()

        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Episode {ep}  Loss {loss.item():.3f}  Return {sum(rewards):.3f}")


def train_mapping_agent(
    array1, array2, big_array3,
    bins=11, episodes=2000, gamma=0.99,
    hidden_dim=128, lr=3e-4, device='cpu'
):
    env = MappingEnv(array1, array2, big_array3, bins=bins)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    train_reinforce(env, policy, optimizer, episodes=episodes, gamma=gamma)
    return policy


def get_mapping_matrix(policy, array1, array2, big_array3, bins=11):
    env = MappingEnv(array1, array2, big_array3, bins=bins)
    N, M = len(array1), len(big_array3)
    alloc = np.zeros((N, M), dtype=float)

    obs = env.reset()
    done = False
    step = 0

    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        logits = policy(obs_t)
        a = torch.distributions.Categorical(logits=logits).sample().item()

        # decode and clamp
        raw_i = a // bins
        b     = a % bins
        x     = b / (bins - 1)
        i     = max(raw_i, env.last_i)
        i_next= min(i+1, M-1)

        alloc[step, i]      = x
        alloc[step, i_next] = 1 - x

        obs, _, done, _ = env.step(a)
        step += 1

    # final reward can be computed if needed
    return alloc


# Example run
if __name__ == "__main__":
    array1     = [-0.1, -0.2, -0.15, -0.05]*5
    array2     = [100,   200,   150,   50]*5
    big_array3 = [1, 2, 3, 5, 7, 10, 20, 30]

    policy = train_mapping_agent(array1, array2, big_array3,
                                 bins=11, episodes=1000)
    mapping = get_mapping_matrix(policy, array1, array2, big_array3, bins=11)
    print("Final mapping matrix:\n", mapping)
    
    def compute_exposure_loss(array1, array2, durations, X):
        """
        Compute the Euclidean norm of the net‐exposure per bucket:
        mismatch_i = sum_j [ array2[j]*durations[i]/10000 + array1[j] ] * X[j,i]
        Loss = || mismatch ||_2
        """
        A1 = np.asarray(array1, dtype=float)           # (N,)
        A2 = np.asarray(array2, dtype=float)           # (N,)
        d  = np.asarray(durations, dtype=float)        # (M,)
        N, M = X.shape
        assert A1.shape[0] == N and A2.shape[0] == N and d.shape[0] == M

        # build coefficient matrix C of shape (N,M) where
        # C[j,i] = array2[j]*durations[i]/10000 + array1[j]
        C = A2[:,None] * (d[None,:]/10000.0) + A1[:,None]  # (N,M)

        # net exposure per bucket
        mismatch = np.sum(C * X, axis=0)  # (M,)

        # L2‐norm
        loss = np.linalg.norm(mismatch, ord=2)
        return loss
    
    loss = compute_exposure_loss(array1, array2, big_array3, mapping)
    print("L2 exposure‐mismatch loss:", loss)