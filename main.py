import numpy as np
import torch
import torch.nn as nn
import gym

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
    def __init__(self, input_dim, reservoir_dim, output_dim, device, symbolic_dim=2, spectral_radius=0.9, sparsity=0.1):
        super(NeuroSymbolicEchoStateNetwork, self).__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.symbolic_dim = symbolic_dim
        self.device = device

        self.Win = torch.randn(reservoir_dim, input_dim, device=device) * 0.1

        W = torch.randn(reservoir_dim, reservoir_dim)
        W *= (np.random.rand(*W.shape) < sparsity)
        eigs = np.max(np.abs(np.linalg.eigvals(W)))
        W = W / eigs * spectral_radius
        self.W = torch.tensor(W, dtype=torch.float32, device=device).clone().detach()

        self.state = torch.zeros(reservoir_dim, dtype=torch.float32, device=device)

        self.symbolic_module = SymbolicReasoningModule(device=device)

        self.readout = nn.Linear(reservoir_dim + symbolic_dim, output_dim).to(device)

    def forward(self, x):
        self.state = torch.tanh(
            torch.matmul(self.Win, x) + torch.matmul(self.W, self.state)
        )

        symbolic_output = self.symbolic_module.forward(x)

        combined_input = torch.cat((self.state, symbolic_output))

        output = self.readout(combined_input)
        return output

    def refine_symbolic_rules(self, feedback):
        self.symbolic_module.refine_rules(feedback)

class PolicyNetwork(nn.Module):
    def __init__(self, esn, action_space):
        super(PolicyNetwork, self).__init__()
        self.esn = esn
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.esn(x)
        probabilities = self.softmax(logits)
        return probabilities

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reservoir_dim = 200

    esn = NeuroSymbolicEchoStateNetwork(
        input_dim=input_dim,
        reservoir_dim=reservoir_dim,
        output_dim=action_dim,
        device=device
    )
    policy = PolicyNetwork(esn, action_dim).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.99

    for episode in range(700):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        rewards = []
        log_probs = []
        feedback = {"pole_angle": 0, "cart_position": 0}

        done = False
        while not done:
            action_probs = policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            rewards.append(reward)

            feedback["pole_angle"] += reward if next_state[2].abs() < 0.1 else -1
            feedback["cart_position"] += reward if next_state[0].abs() < 1 else -1

            state = next_state

        esn.refine_symbolic_rules(feedback)

        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        loss = 0
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss += -log_prob * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.close()

if __name__ == "__main__":
    train()
