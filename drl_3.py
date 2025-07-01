"""Trying to use deep RL (ppo, because stable & not too crazy) to do cashflow allocation for dv01 matching thing.  
like, have a bunch of liab cashflows, each with pv and a dv01, and then some duration buckets (e.g. bonds with 2y, 5y, whatever),  
the goal is allocate (split) each cf across TWO neighboring buckets, so that total dv01 per bucket ≈ liab dv01s (as close as poss).  
Fractional allowed, but must fill each cf exactly, & only split between 2 buckets at a time.  
Loss is mean squared error of dv01 per bucket, between “hedged” and “target” dv01.

Approach
--------
- treat it as a sequential decison process, so step by step:  
    for each cf, pick where to split (choose bucket idx), and how much in the first of the pair (fraction, [0,1])
    after each step, update state & get reward = drop in mse (old-new), so agent learns to make things better each move  
    finish when all cfs done

- policy net is a basic MLP in pytorch (overkill maybe but easy), takes state:
    - outputs logits for allowed buckets (categorical action), and a sigmoid (fraction)
    - only the allowed pairs (never backtrack buckets), so last bucket gets updated
- train with PPO, including strong entropy reg (needed or it gets stuck on dumb solutions)
- run greedy policy at the end to see what the agent learned

Code
----
- CashflowAllocationEnv: env logic, state, step, mse calc, tracking what’s left etc
- PolicyNet: takes state, outputs bucket idx and split fraction, standard torch style
- RolloutBuffer: just collects states/actions for PPO update, nothing fancy
- compute_returns: normal RL trick, discount rewards
- train_ppo: main loop, sample rollout, train, print mse every now & then
- __main__ block: runs a small test (tiny problem for demo), runs training, then prints matrix, row sums, dv01 etc,  
  also shows what the “correct” answer would be for sanity check

Notes
-----
- reward = prev_mse - curr_mse, so it’s always positive for “good” moves (hopefully helps with stability)
- state includes progress (step idx, last bucket, what’s left)
- only ever split across two buckets, never more, because too much headache otherwise lol
- maybe could vectorize more, but honestly this is just for learning
- not super clean or perf, just gets the job done.  
- i think real apps would need more constrs but this shows the core idea
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CashflowAllocationEnv:
    def __init__(self, array1, array2, durations):
        self.N = len(array1)
        self.M = len(durations)
        self.array1 = np.array(array1, dtype=np.float32)
        self.array2 = np.array(array2, dtype=np.float32)
        self.durations = np.array(durations, dtype=np.float32)
        self.reset()

    def reset(self):
        self.actions_taken = []
        self.step_idx = 0
        self.last_bucket = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        s = []
        s.append(self.step_idx / self.N)
        s.append(self.last_bucket / (self.M - 1))
        rem_dv01 = np.zeros(self.N)
        rem_pv = np.zeros(self.N)
        if self.step_idx < self.N:
            rem_dv01[:self.N - self.step_idx] = self.array1[self.step_idx:]
            rem_pv[:self.N - self.step_idx] = self.array2[self.step_idx:]
        s.extend(rem_dv01)
        s.extend(rem_pv)
        return np.array(s, dtype=np.float32)

    def get_valid_buckets(self):
        return np.arange(self.last_bucket, self.M - 1)

    def get_allocation(self):
        X = np.zeros((self.N, self.M), dtype=np.float32)
        for j, (i_j, x_j) in enumerate(self.actions_taken):
            X[j, i_j] = x_j
            X[j, i_j + 1] = 1.0 - x_j
        return X

    def _compute_mse(self, X):
        hedged_dv01 = (X * self.array2[:, None]).sum(axis=0) * (self.durations / 10000)
        liab_dv01 = (X * self.array1[:, None]).sum(axis=0)
        mse = np.mean((hedged_dv01 - liab_dv01) ** 2)
        return mse

    def step(self, action):
        assert not self.done
        j = self.step_idx
        i = int(action[0])
        x = float(np.clip(action[1], 0.0, 1.0))
        valid = self.get_valid_buckets()
        assert i in valid, f"Bucket {i} invalid at step {j} (allowed: {valid})"
        X = np.zeros((self.N, self.M), dtype=np.float32)
        for k, (bk, xk) in enumerate(self.actions_taken):
            X[k, bk] = xk
            X[k, bk + 1] = 1.0 - xk
        X[j, i] = x
        X[j, i+1] = 1-x
        prev_mse = self._compute_mse(X) if j > 0 else 0.0
        self.actions_taken.append((i, x))
        self.step_idx += 1
        self.last_bucket = i
        self.done = (self.step_idx == self.N)
        X_new = self.get_allocation()
        curr_mse = self._compute_mse(X_new)
        reward = prev_mse - curr_mse  # Positive if improvement
        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def get_final_mse(self):
        X = self.get_allocation()
        return self._compute_mse(X)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, max_buckets):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.bucket_head = nn.Linear(128, max_buckets)
        self.frac_head = nn.Linear(128, 1)

    def forward(self, state, num_valid):
        h = self.shared(state)
        bucket_logits = self.bucket_head(h)[:, :num_valid]
        split_frac = torch.sigmoid(self.frac_head(h)).squeeze(-1)
        return bucket_logits, split_frac

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.bucket_idxs = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def store(self, s, a, bucket_idx, lp, r, d):
        self.states.append(s)
        self.actions.append(a)
        self.bucket_idxs.append(bucket_idx)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.dones.append(d)

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.bucket_idxs, dtype=torch.long),
            torch.stack(self.logprobs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32)
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.bucket_idxs.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()

def compute_returns(rewards, gamma=0.98):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def train_ppo(array1, array2, durations, epochs=2000, lr=1e-3, gamma=0.98, clip_eps=0.2, update_steps=5, print_every=100):
    N, M = len(array1), len(durations)
    dummy_env = CashflowAllocationEnv(array1, array2, durations)
    state_dim = dummy_env._get_state().shape[0]
    del dummy_env

    policy = PolicyNet(state_dim, max_buckets=M-1)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer()

    for ep in range(epochs):
        env = CashflowAllocationEnv(array1, array2, durations)
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            valid_buckets = env.get_valid_buckets()
            num_valid = len(valid_buckets)
            bucket_logits, split_frac = policy(s, num_valid)
            bucket_dist = torch.distributions.Categorical(logits=bucket_logits)
            bucket_idx = bucket_dist.sample()
            chosen_bucket = valid_buckets[bucket_idx.item()]
            # Add more noise for split fraction exploration!
            noisy_split = (split_frac + 0.2 * torch.randn_like(split_frac)).clamp(0, 1)
            action_tensor = torch.tensor([chosen_bucket, noisy_split.item()])
            logprob = bucket_dist.log_prob(bucket_idx) + torch.log(
                torch.clamp(noisy_split, 1e-8, 1 - 1e-8)
            )
            next_state, reward, done, _ = env.step((chosen_bucket, noisy_split.item()))
            buffer.store(s.squeeze(0), action_tensor, bucket_idx.item(), logprob, reward, done)
            state = next_state
            ep_reward += reward

        # PPO update
        states, actions, bucket_idxs, logprobs, rewards, dones = buffer.get()
        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(update_steps):
            num_valids = [min(M-1, M-1-int(states[i][1].item()*(M-1))) for i in range(len(states))]
            max_valid = max(num_valids)
            bucket_logits_list = []
            split_frac_list = []
            for i in range(len(states)):
                s = states[i].unsqueeze(0)
                bucket_logits, split_frac = policy(s, num_valids[i])
                pad = torch.full((1, max_valid - num_valids[i]), -1e9)
                bucket_logits = torch.cat([bucket_logits, pad], dim=1)
                bucket_logits_list.append(bucket_logits)
                split_frac_list.append(split_frac)
            batch_bucket_logits = torch.cat(bucket_logits_list, dim=0)
            batch_split_frac = torch.stack(split_frac_list).squeeze(-1)
            bucket_dist = torch.distributions.Categorical(logits=batch_bucket_logits)
            logprob_buckets = bucket_dist.log_prob(bucket_idxs)
            logprob_fracs = torch.log(torch.clamp(batch_split_frac, 1e-8, 1 - 1e-8))
            logprobs_new = logprob_buckets + logprob_fracs
            ratios = torch.exp(logprobs_new - logprobs.detach())
            advantages = returns
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            # --- Strong entropy regularization for exploration ---
            entropy_buckets = bucket_dist.entropy().mean()
            entropy_fracs = -(batch_split_frac * torch.log(batch_split_frac + 1e-8) +
                              (1-batch_split_frac) * torch.log(1-batch_split_frac + 1e-8)).mean()
            entropy_coeff = 0.15
            loss = -torch.min(surr1, surr2).mean() - entropy_coeff * (entropy_buckets + entropy_fracs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % print_every == 0 or ep == epochs - 1:
            mse = env.get_final_mse()
            print(f"Episode {ep}: Final MSE {mse:.8f}")
        buffer.clear()
    return policy

if __name__ == "__main__":
    N, M = 3, 4
    durations = np.array([2, 5, 10, 30], dtype=np.float32)
    array2 = np.array([100, 200, 300], dtype=np.float32)
    array1 = np.array([-0.02, -0.10, -0.30], dtype=np.float32)  # matches the hedged DV01 for each

    print("PVs:     ", array2)
    print("LiabDV01:", array1)
    print("Durations:", durations)

    policy = train_ppo(array1, array2, durations, epochs=500, print_every=100)

    env = CashflowAllocationEnv(array1, array2, durations)
    state = env.reset()
    done = False
    while not done:
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        valid_buckets = env.get_valid_buckets()
        num_valid = len(valid_buckets)
        bucket_logits, split_frac = policy(s, num_valid)
        bucket_idx = bucket_logits.argmax().item()
        chosen_bucket = valid_buckets[bucket_idx]
        state, _, done, _ = env.step((chosen_bucket, split_frac.item()))

    X = env.get_allocation()
    print("\nFinal allocation matrix (N x M):")
    print(np.round(X, 4))
    print("Row sums:", X.sum(axis=1))

    print("\nCashflow allocations:")
    for j, (i_j, x_j) in enumerate(env.actions_taken):
        print(f"CF{j}: bucket {i_j} -> {x_j:.4f}, bucket {i_j+1} -> {1-x_j:.4f}")

    hedged_dv01 = (X * array2[:, None]).sum(axis=0) * (durations / 10000)
    liab_dv01 = (X * array1[:, None]).sum(axis=0)
    print("\nPer-bucket hedged DV01:   ", np.round(hedged_dv01, 4))
    print("Per-bucket liability DV01:", np.round(liab_dv01, 4))
    print("Diff:", np.round(hedged_dv01 - liab_dv01, 8))
    print("Final MSE:", env.get_final_mse())

    # Show known correct allocation for comparison
    X_true = np.eye(N, M)
    print("\nKnown correct allocation (rows):")
    print(X_true)
