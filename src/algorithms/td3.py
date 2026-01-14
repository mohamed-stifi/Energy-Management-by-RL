"""
Twin Delayed DDPG (TD3) - Online RL Algorithm.

Reference: Fujimoto et al., 2018
https://arxiv.org/abs/1802.09477
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import os


class ReplayBuffer:
    """Experience replay buffer."""
    
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
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.action_low = torch.FloatTensor(action_low)
        self.action_scale = (torch.FloatTensor(action_high) - self.action_low) / 2.0
        self.action_bias = (torch.FloatTensor(action_high) + self.action_low) / 2.0

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_tanh = torch.tanh(self.fc3(x))
        action = action_tanh * self.action_scale + self.action_bias
        return action


class Critic(nn.Module):
    """Dual Critic network (two Q-networks)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        q1 = torch.relu(self.fc1_q1(x))
        q1 = torch.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        q2 = torch.relu(self.fc1_q2(x))
        q2 = torch.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2


class TD3:
    """Twin Delayed DDPG Algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_std: float = 0.2,
        device: str = "cpu",
        action_low: np.ndarray = np.array([12., 23.25, -1.]),
        action_high: np.ndarray = np.array([23.25, 30., 1.])
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.device = torch.device(device)
        self.update_step = 0
        
        # ⭐ Stocker les bornes pour select_action
        self.action_low = torch.FloatTensor(action_low).to(self.device)
        self.action_high = torch.FloatTensor(action_high).to(self.device)
        
        # ⭐ Networks avec scaling
        self.actor = Actor(
            state_dim, action_dim, hidden_dim,
            action_low=action_low, action_high=action_high
        ).to(self.device)
        
        self.target_actor = Actor(
            state_dim, action_dim, hidden_dim,
            action_low=action_low, action_high=action_high
        ).to(self.device)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights
        self._copy_weights(self.actor, self.target_actor)
        self._copy_weights(self.critic, self.target_critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
    
    def _copy_weights(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Select action from policy."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actor(state_tensor)
                
                if not deterministic:
                    noise = torch.FloatTensor(
                        np.random.normal(0, self.noise_std, self.action_dim)
                    ).to(self.device)
                    action = action + noise
                
                # ⭐ Clamp avec les bornes stockées dans TD3
                action = torch.clamp(action, self.action_low, self.action_high)
            
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
        
        # Compute target Q
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_noise = torch.FloatTensor(
                np.random.normal(0, self.noise_std, actions.shape)
            ).to(self.device)
            next_actions = torch.clamp(next_actions + next_noise, -1, 1)
            
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q
        
        # Update Critic
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy update
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic, self.target_critic)
        
        self.update_step += 1
        
        return {'critic_loss': critic_loss.item()}
    
    def save(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])