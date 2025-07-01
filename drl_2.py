#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import spaces

# --- Environment ---
class MappingEnv(gym.Env):
    """
    Gym environment: at each step j, choose a bucket i and fraction x.
    Action space: Tuple(Discrete(M), Box(0,1))
    """
    metadata = {'render.modes': []}

    def __init__(self, array1, array2, durations):
        super().__init__()
        self.array1 = np.asarray(array1, dtype=float)
        self.array2 = np.asarray(array2, dtype=float)
        self.durs = np.asarray(durations, dtype=float) / 10000.0
        self.N = len(self.array1)
        self.M = len(self.durs)

        # actions: i in [0,M-1], x in [0,1]
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.M),
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        ))
        # obs: [j/N, resid_DV01, resid_PV, last_bucket/M]
        low = np.array([0.0, -1e9, -1e9, 0.0], dtype=float)
        high = np.array([1.0, 1e9, 1e9, 1.0], dtype=float)
        self.observation_space = spaces.Box(low, high, dtype=float)
        self.reset()

    def reset(self):
        self.j = 0
        self.resid_DV01 = 0.0
        self.resid_PV = 0.0
        self.last_bucket = 0
        return self._get_obs()

    def step(self, action):
        i, x_arr = action
        x = float(x_arr[0])
        # enforce monotonicity
        i = max(i, self.last_bucket)
        i_next = min(i+1, self.M-1)

        dv01 = self.array1[self.j]
        pv = self.array2[self.j]
        hedged_dv01 = x*(dv01 + pv*self.durs[i]) + (1-x)*(dv01 + pv*self.durs[i_next])
        hedged_pv = x*pv + (1-x)*pv

        prev_err = self.resid_DV01**2 + self.resid_PV**2
        self.resid_DV01 += (dv01 - hedged_dv01)
        self.resid_PV   += (pv   - hedged_pv)
        curr_err = self.resid_DV01**2 + self.resid_PV**2

        # reward = reduction in squared error
        reward = prev_err - curr_err

        self.last_bucket = i_next
        self.j += 1
        done = (self.j >= self.N)
        if done:
            # terminal bonus of negative final error
            reward += -curr_err

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([
            self.j / self.N,
            self.resid_DV01,
            self.resid_PV,
            self.last_bucket / max(1, self.M-1)
        ], dtype=float)

# --- Actor-Critic ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, M):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        # discrete head
        self.logits = nn.Linear(128, M)
        # continuous head: Beta parameters
        self.alpha  = nn.Linear(128, 1)
        self.beta   = nn.Linear(128, 1)
        # value head
        self.value  = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.logits(h)
        # ensure positivity
        alpha = F.softplus(self.alpha(h)) + 1e-3
        beta  = F.softplus(self.beta(h))  + 1e-3
        value = self.value(h).squeeze(-1)
        return logits, alpha.squeeze(-1), beta.squeeze(-1), value

# --- PPO Training ---
def ppo_train(env, policy, optimizer,
              epochs=200, timesteps=1024,
              clip_eps=0.2, gamma=0.99, lam=0.95,
              vf_coef=0.5, ent_coef=0.01):
    obs_dim = env.observation_space.shape[0]
    M = env.M

    # buffers
    obs_buf = np.zeros((timesteps, obs_dim), dtype=float)
    act_i_buf = np.zeros(timesteps, dtype=int)
    act_x_buf = np.zeros((timesteps,1), dtype=float)
    rew_buf = np.zeros(timesteps, dtype=float)
    val_buf = np.zeros(timesteps, dtype=float)
    logp_buf = np.zeros(timesteps, dtype=float)

    for epoch in range(epochs):
        obs = env.reset()
        for t in range(timesteps):
            obs_buf[t] = obs
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits, alpha, beta, value = policy(obs_t)
                dist_i = torch.distributions.Categorical(logits=logits)
                dist_x = torch.distributions.Beta(alpha, beta)
                ai = dist_i.sample().item()
                ax = dist_x.sample().item()
                logp = dist_i.log_prob(torch.tensor(ai)) + dist_x.log_prob(torch.tensor(ax))
            act_i_buf[t] = ai
            act_x_buf[t] = ax
            val_buf[t] = value.item()
            logp_buf[t] = logp.item()

            obs, rew, done, _ = env.step((ai, np.array([ax])))
            rew_buf[t] = rew
            if done:
                obs = env.reset()

        # last value for GAE
        with torch.no_grad():
            _, _, _, last_val = policy(torch.from_numpy(obs).float().unsqueeze(0))
        # GAE
        adv_buf = np.zeros_like(rew_buf)
        lastgaelam = 0
        for t in reversed(range(timesteps)):
            if t == timesteps - 1:
                nextnonterm = 1.0
                nextval = last_val.item()
            else:
                nextnonterm = 1.0
                nextval = val_buf[t+1]
            delta = rew_buf[t] + gamma * nextnonterm * nextval - val_buf[t]
            adv_buf[t] = lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
        ret_buf = adv_buf + val_buf

        # to tensors
        obs_t = torch.from_numpy(obs_buf).float()
        act_i_t = torch.from_numpy(act_i_buf)
        act_x_t = torch.from_numpy(act_x_buf).float().squeeze(-1)
        old_val_t = torch.from_numpy(val_buf).float()
        old_logp_t = torch.from_numpy(logp_buf).float()
        ret_t = torch.from_numpy(ret_buf).float()
        adv_t = torch.from_numpy(adv_buf).float()

        # PPO update
        for _ in range(4):
            logits, alpha, beta, value = policy(obs_t)
            dist_i = torch.distributions.Categorical(logits=logits)
            dist_x = torch.distributions.Beta(alpha, beta)
            logp = dist_i.log_prob(act_i_t) + dist_x.log_prob(act_x_t)
            ratio = torch.exp(logp - old_logp_t)
            pg_loss = -torch.mean(torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*adv_t
            ))
            v_loss = torch.mean((value - ret_t)**2)
            ent = torch.mean(dist_i.entropy() + dist_x.entropy())
            loss = pg_loss + vf_coef*v_loss - ent_coef*ent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3} loss {loss.item():.3f} return {rew_buf.sum():.3f}")

# --- Main ---
if __name__ == "__main__":
    array1 = [-0.1, -0.2, -0.15, -0.05]
    array2 = [100,   200,   150,    50]
    big_array3 = [1, 2, 3, 5, 7, 10, 20, 30]

    env = MappingEnv(array1, array2, big_array3)
    obs_dim = env.observation_space.shape[0]
    M = env.M

    policy = ActorCritic(obs_dim, M)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    ppo_train(env, policy, optimizer)

    # Extract allocation matrix
    alloc = np.zeros((env.N, env.M), dtype=float)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        logits, alpha, beta, _ = policy(obs_t)
        ai = torch.distributions.Categorical(logits=logits).sample().item()
        ax = torch.distributions.Beta(alpha, beta).sample().item()
        i = max(ai, env.last_bucket)
        inext = min(i+1, env.M-1)
        alloc[step, i] = ax
        alloc[step, inext] = 1-ax
        obs, _, done, _ = env.step((ai, np.array([ax])))
        step += 1

    print("Allocation matrix:\n", np.round(alloc,4))

import numpy as np

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


array1     = [-0.1, -0.2, -0.15, -0.05]
array2     = [100,   200,   150,    50]
durations  = [1, 2, 3, 5, 7, 10, 20, 30]


loss = compute_exposure_loss(array1, array2, durations, X)
print("L2 exposure‐mismatch loss:", loss)
