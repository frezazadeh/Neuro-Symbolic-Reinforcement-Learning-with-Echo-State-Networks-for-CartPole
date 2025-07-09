import numpy as np
# alias the old numpy bool8 name so Gym's checker won't blow up
np.bool8 = np.bool_

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import gym

# optional
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()


class SymbolicReasoningModule:
    def __init__(self, device):
        self.device = device
        self.rules = {
            "pole_angle": lambda angle: [0.9, 0.1] if angle < -0.1 else ([0.1, 0.9] if angle > 0.1 else [0.5, 0.5]),
            "cart_position": lambda pos: [0.8, 0.2] if pos < -1 else ([0.2, 0.8] if pos > 1 else [0.5, 0.5])
        }

    def forward(self, state):
        pole_angle = state[2].item()
        cart_position = state[0].item()
        angle_output = self.rules["pole_angle"](pole_angle)
        position_output = self.rules["cart_position"](cart_position)
        symbolic_output = [(a + b) / 2 for a, b in zip(angle_output, position_output)]
        return torch.tensor(symbolic_output, dtype=torch.float32, device=self.device)

    def refine_rules(self, feedback):
        for key in self.rules:
            if feedback[key] < 0:
                self.rules[key] = lambda x: [0.6, 0.4] if x < -0.1 else ([0.4, 0.6] if x > 0.1 else [0.5, 0.5])


class NeuroSymbolicEchoStateNetwork(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, device,
                 symbolic_dim=2, spectral_radius=0.9, sparsity=0.1):
        super().__init__()
        self.device = device
        self.reservoir_dim = reservoir_dim
        self.symbolic_dim = symbolic_dim

        # input-to-reservoir
        self.Win = torch.randn(reservoir_dim, input_dim, device=device) * 0.1

        # reservoir recurrent weights (sparse + scaled)
        W = np.random.randn(reservoir_dim, reservoir_dim)
        W *= (np.random.rand(reservoir_dim, reservoir_dim) < sparsity)
        eigs = np.max(np.abs(np.linalg.eigvals(W)))
        W = W / eigs * spectral_radius
        # use from_numpy to avoid the copy‐construct warning
        self.W = torch.from_numpy(W.astype(np.float32)).to(device)

        self.state = torch.zeros(reservoir_dim, device=device, dtype=torch.float32)
        self.symbolic_module = SymbolicReasoningModule(device=device)
        self.readout = nn.Linear(reservoir_dim + symbolic_dim, output_dim).to(device)

    def forward(self, x):
        self.state = torch.tanh(self.Win @ x + self.W @ self.state)
        symbolic_output = self.symbolic_module.forward(x)
        combined = torch.cat((self.state, symbolic_output))
        return self.readout(combined)

    def refine_symbolic_rules(self, feedback):
        self.symbolic_module.refine_rules(feedback)


class PolicyNetwork(nn.Module):
    def __init__(self, esn, action_space):
        super().__init__()
        self.esn = esn
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.esn(x))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", new_step_api=True)

    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reservoir_dim = 150

    esn = NeuroSymbolicEchoStateNetwork(input_dim, reservoir_dim, action_dim, device)
    policy = PolicyNetwork(esn, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99

    for episode in range(500):
        # reset may return obs or (obs, info, ...)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        state = torch.tensor(obs, dtype=torch.float32, device=device)

        rewards, log_probs = [], []
        feedback = {"pole_angle": 0, "cart_position": 0}
        done = False

        while not done:
            probs = policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            next_out = env.step(action.item())
            obs, reward, terminated, truncated, _ = next_out
            done = terminated or truncated

            next_state = torch.tensor(obs, dtype=torch.float32, device=device)
            rewards.append(reward)
            feedback["pole_angle"] += reward if abs(next_state[2].item()) < 0.1 else -1
            feedback["cart_position"] += reward if abs(next_state[0].item()) < 1 else -1

            state = next_state

        esn.refine_symbolic_rules(feedback)

        # compute discounted returns
        discounted, R = [], 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)
        disc = torch.tensor(discounted, dtype=torch.float32, device=device)
        disc = (disc - disc.mean()) / (disc.std() + 1e-8)

        # policy‐gradient loss
        loss = sum(-lp * R for lp, R in zip(log_probs, disc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode:03d} — Total Reward: {sum(rewards)}")

    env.close()


if __name__ == "__main__":
    train()
