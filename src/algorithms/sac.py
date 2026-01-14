"""
Soft Actor-Critic (SAC) - Online RL Algorithm.

Reference: Haarnoja et al., 2018
https://arxiv.org/abs/1801.01290
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import os


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, max_size: int = 1e6):
        self.buffer = deque(maxlen=int(max_size))
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_low: np.ndarray = np.array([12., 23.25, -1.]),
                 action_high: np.ndarray = np.array([23.25, 30., 1.])):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Bornes réelles de l'action space
        self.action_low = torch.FloatTensor(action_low)
        self.action_high = torch.FloatTensor(action_high)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        z = torch.randn_like(std)
        action = mean + std * z
        
        # Appliquer tanh puis scaling aux bornes réelles
        action_tanh = torch.tanh(action)
        scaled_action = action_tanh * self.action_scale + self.action_bias
        
        log_prob = -0.5 * (z**2 + np.log(2 * np.pi)) - log_std - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, mean, log_std


class Critic(nn.Module):
    """Critic network for SAC (Q-function)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class SAC:
    """Soft Actor-Critic Algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device) 
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim,
                   action_low=np.array([12., 23.25, -1.]),
                   action_high=np.array([23.25, 30., 1.]))
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self._copy_weights(self.critic1, self.target_critic1)
        self._copy_weights(self.critic2, self.target_critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
    
    def _copy_weights(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = mean
            else:
                action, _, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy().squeeze()
    
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        """One training step."""
        if len(replay_buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, next_means, next_log_stds = self.actor.sample(next_states)
            next_log_probs = -next_log_stds.sum(dim=1, keepdim=True) - 0.5 * np.log(2 * np.pi) * self.action_dim
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) + self.alpha * next_log_probs
            
            target_q = rewards + (1.0 - dones) * self.gamma * target_q
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = ((q1 - target_q) ** 2).mean()
        critic2_loss = ((q2 - target_q) ** 2).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update Actor
        actions_sample, means, log_stds = self.actor.sample(states)
        log_probs = -log_stds.sum(dim=1, keepdim=True) - 0.5 * np.log(2 * np.pi) * self.action_dim
        
        q1 = self.critic1(states, actions_sample)
        q2 = self.critic2(states, actions_sample)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])