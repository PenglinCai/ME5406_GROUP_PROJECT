import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from copy import deepcopy
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

# ---------------- Exploration Noise ----------------
class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.05, decay=0.995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state

    def decay_sigma(self):
        self.sigma = max(self.sigma * self.decay, self.sigma_min)

# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)

    def __len__(self):
        return len(self.buffer)

# ---------------- Neural Networks ----------------
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, a_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.out(x)

# ---------------- DDPG Agent ----------------
class DDPG_OU_Agent:
    def __init__(
        self, state_dim, action_dim, action_range,
        buffer_capacity=50000, gamma=0.99, tau=0.005,
        actor_lr=1e-4, critic_lr=1e-3, hidden_dim=256,
        noise_theta=0.15, noise_sigma=0.2
    ):
        self.s_dim, self.a_dim = state_dim, action_dim
        try:
            self.action_range = float(action_range[1])
        except:
            self.action_range = float(action_range)
        self.gamma, self.tau = gamma, tau

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # actor / critic + target networks
        self.actor         = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic        = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor  = deepcopy(self.actor).to(device)
        self.target_critic = deepcopy(self.critic).to(device)

        # weight initialization
        self._init_weights(self.actor)
        self._init_weights(self.critic)

        # optimizers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # exploration noise process
        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)

    def _init_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def select_action(self, state, evaluate=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        raw_action = self.actor(s).cpu().detach().numpy()[0]
        action = raw_action * self.action_range
        if not evaluate:
            noise = self.noise() * self.action_range
            action = action + noise
        return np.clip(action, -self.action_range, self.action_range)

    def reset_noise(self):
        self.noise.reset()

    def store_transition(self, s, a, r, s2, done):
        self.replay_buffer.push(s, a, r, s2, done)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s  = torch.FloatTensor(s).to(device)
        a  = torch.FloatTensor(a).to(device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(device)
        s2 = torch.FloatTensor(s2).to(device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            a2 = self.target_actor(s2) * self.action_range
            q2 = self.target_critic(s2, a2)
            q_target = r + (1 - d) * self.gamma * q2
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # Actor update
        a_pred = self.actor(s) * self.action_range
        actor_loss = - self.critic(s, a_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # Soft update
        for tgt, src in zip(self.target_actor.parameters(),  self.actor.parameters()):
            tgt.data.copy_( tgt.data * (1 - self.tau) + src.data * self.tau )
        for tgt, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            tgt.data.copy_( tgt.data * (1 - self.tau) + src.data * self.tau )

        # decay exploration noise
        self.noise.decay_sigma()

        return actor_loss.item(), critic_loss.item()

    def save(self, save_dir="Models_DDPG_OU"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

    def load(self, save_dir="Models_DDPG_OU"):
        self.actor.load_state_dict(torch.load(os.path.join(save_dir,"actor.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(save_dir,"critic.pth"), map_location=device))
        # sync target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def evaluate_q(self, state, action):
        self.critic.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            a = torch.FloatTensor(action).unsqueeze(0).to(device)
            q_val = self.critic(s, a).item()
        self.critic.train()
        return q_val
