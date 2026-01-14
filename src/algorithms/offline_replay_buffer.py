"""
Offline Replay Buffer for storing and managing offline RL data.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


class OfflineReplayBuffer:
    """
    Buffer for storing transitions from offline episodes.
    Transitions: (obs, action, reward, next_obs, done, info)
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.infos = []
        self.size = 0
    
    def add(self, obs, action, reward, next_obs, done, info=None):
        """Add a single transition."""
        if self.size >= self.max_size:
            # Remove oldest transition
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_observations.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)
        else:
            self.size += 1
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.infos.append(info if info is not None else {})
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        if len(self.observations) < batch_size:
            raise ValueError(f"Buffer has {len(self.observations)} < batch_size {batch_size}")
        
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        return (
            np.array([self.observations[i] for i in indices]),
            np.array([self.actions[i] for i in indices]),
            np.array([self.rewards[i] for i in indices]).reshape(-1, 1),
            np.array([self.next_observations[i] for i in indices]),
            np.array([self.dones[i] for i in indices]).reshape(-1, 1),
        )
    
    def get_all(self) -> Tuple:
        """Return all transitions."""
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards).reshape(-1, 1),
            np.array(self.next_observations),
            np.array(self.dones).reshape(-1, 1),
        )
    
    def save(self, filepath: str):
        """Save buffer to pickle file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_observations': self.next_observations,
            'dones': self.dones,
            'infos': self.infos,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Buffer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_observations = data['next_observations']
        self.dones = data['dones']
        self.infos = data['infos']
        self.size = len(self.observations)
        print(f"✅ Buffer loaded from {filepath} ({self.size} transitions)")
    
    def get_statistics(self) -> Dict:
        """Compute statistics of the buffer."""
        rewards = np.array(self.rewards).flatten()
        obs_array = np.array(self.observations)
        actions_array = np.array(self.actions)
        
        stats = {
            'num_transitions': len(self.observations),
            'reward_mean': float(rewards.mean()),
            'reward_std': float(rewards.std()),
            'reward_min': float(rewards.min()),
            'reward_max': float(rewards.max()),
            'obs_shape': obs_array.shape[1:],
            'action_shape': actions_array.shape[1:],
            'obs_mean': obs_array.mean(axis=0).tolist(),
            'obs_std': obs_array.std(axis=0).tolist(),
            'action_mean': actions_array.mean(axis=0).tolist(),
            'action_std': actions_array.std(axis=0).tolist(),
        }
        return stats
    
    def __len__(self):
        return len(self.observations)


def save_statistics(stats: Dict, filepath: str):
    """Save statistics to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved to {filepath}")


def load_statistics(filepath: str) -> Dict:
    """Load statistics from JSON."""
    with open(filepath, 'r') as f:
        stats = json.load(f)
    print(f"✅ Statistics loaded from {filepath}")
    return stats