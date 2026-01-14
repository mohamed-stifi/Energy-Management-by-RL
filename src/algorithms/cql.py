"""
Conservative Q-Learning (CQL) - Offline RL Algorithm avec Double Q-Learning.

Reference: Kumar et al., 2020
https://arxiv.org/abs/2006.04779
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import os


class OfflineReplayBuffer:
    """Offline replay buffer (no sampling with replacement)."""
    
    def __init__(self, max_size: int = 1e6):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.max_size = int(max_size)
    
    def add(self, state, action, reward, next_state, done):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.actions[i] for i in indices]),
            np.array([self.rewards[i] for i in indices]).reshape(-1, 1),
            np.array([self.next_states[i] for i in indices]),
            np.array([self.dones[i] for i in indices]).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.states)


class CriticQ(nn.Module):
    """Q-network for CQL."""
    
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


class CQL:
    """Conservative Q-Learning (Offline RL) avec Double Q-Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        cql_weight: float = 10.0,  # ⬆️ AUGMENTÉ de 1.0 à 10.0
        cql_temp: float = 0.3,      # ⬇️ DIMINUÉ de 1.0 à 0.3
        num_random_actions: int = 10,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_weight = cql_weight
        self.cql_temp = cql_temp
        self.num_random_actions = num_random_actions
        self.device = torch.device(device)
        
        # ⭐ DOUBLE Q-LEARNING : Deux critics au lieu d'un
        self.q_network_1 = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_network_2 = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.target_q_network_1 = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network_2 = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        
        self._copy_weights(self.q_network_1, self.target_q_network_1)
        self._copy_weights(self.q_network_2, self.target_q_network_2)
        
        # Optimizer pour les deux critics
        self.optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=learning_rate)
        self.optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=learning_rate)
    
    def _copy_weights(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, source, target, tau: float = 0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action (CQL is offline, deterministic action)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Évaluer plusieurs actions aléatoires et choisir la meilleure
            best_q = -np.inf
            best_action = None
            
            for _ in range(10):
                action = torch.FloatTensor(np.random.uniform(-1, 1, self.action_dim)).unsqueeze(0).to(self.device)
                # ⭐ Utiliser le min des deux Q-networks pour sélectionner l'action
                q1 = self.q_network_1(state_tensor, action)
                q2 = self.q_network_2(state_tensor, action)
                q = torch.min(q1, q2).item()
                
                if q > best_q:
                    best_q = q
                    best_action = action.cpu().numpy().squeeze()
        
        return best_action if best_action is not None else np.random.uniform(-1, 1, self.action_dim)
    
    def train_step(self, replay_buffer: OfflineReplayBuffer, batch_size: int = 256):
        """One training step with Double Q-Learning + CQL penalty."""
        if len(replay_buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ⭐ DOUBLE Q-LEARNING TARGET : min(Q1_target, Q2_target)
        with torch.no_grad():
            next_q1 = self.target_q_network_1(next_states, actions)
            next_q2 = self.target_q_network_2(next_states, actions)
            next_q = torch.min(next_q1, next_q2)  # ⭐ Min pour réduire overestimation
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        
        # ⭐ UPDATE Q1
        q1_pred = self.q_network_1(states, actions)
        td_loss_1 = ((q1_pred - target_q) ** 2).mean()
        
        # CQL penalty Q1 : pénaliser Q-values sur actions aléatoires
        random_actions = torch.FloatTensor(
            np.random.uniform(-1, 1, (batch_size * self.num_random_actions, self.action_dim))
        ).to(self.device)
        repeated_states = states.repeat(self.num_random_actions, 1)
        q1_random = self.q_network_1(repeated_states, random_actions)
        cql_penalty_1 = torch.logsumexp(q1_random / self.cql_temp, dim=0).mean()
        cql_penalty_1 -= q1_pred.mean()
        
        loss_1 = td_loss_1 + self.cql_weight * cql_penalty_1
        
        self.optimizer_1.zero_grad()
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), 1.0)  # ⭐ Gradient clipping
        self.optimizer_1.step()
        
        # ⭐ UPDATE Q2 (même procédure)
        q2_pred = self.q_network_2(states, actions)
        td_loss_2 = ((q2_pred - target_q) ** 2).mean()
        
        q2_random = self.q_network_2(repeated_states, random_actions)
        cql_penalty_2 = torch.logsumexp(q2_random / self.cql_temp, dim=0).mean()
        cql_penalty_2 -= q2_pred.mean()
        
        loss_2 = td_loss_2 + self.cql_weight * cql_penalty_2
        
        self.optimizer_2.zero_grad()
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_2.parameters(), 1.0)  # ⭐ Gradient clipping
        self.optimizer_2.step()
        
        # Soft update targets
        self._soft_update(self.q_network_1, self.target_q_network_1)
        self._soft_update(self.q_network_2, self.target_q_network_2)
        
        return {
            'td_loss': ((td_loss_1 + td_loss_2) / 2).item(),
            'cql_penalty': ((cql_penalty_1 + cql_penalty_2) / 2).item(),
            'total_loss': ((loss_1 + loss_2) / 2).item(),
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_1': self.q_network_1.state_dict(),
            'q_network_2': self.q_network_2.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        self.q_network_1.load_state_dict(checkpoint['q_network_1'])
        self.q_network_2.load_state_dict(checkpoint['q_network_2'])
        self._copy_weights(self.q_network_1, self.target_q_network_1)
        self._copy_weights(self.q_network_2, self.target_q_network_2)