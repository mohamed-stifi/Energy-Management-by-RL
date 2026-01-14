"""
Helper functions for mixed policies (random + rule-based + expert).
"""
import numpy as np


class RuleBasedPolicy:
    """
    Simple rule-based policy for building control.
    
    Rules:
    1. If price is off-peak AND battery SOC < 80% → charge battery
    2. If price is peak AND battery SOC > 20% → discharge battery
    3. Maintain comfort: adjust HVAC setpoints based on current temperature
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
        # Pré-calculer low/high pour clamp rapide
        self.low = self.action_space.low
        self.high = self.action_space.high

    def get_action(self, obs, info):
        """
        Returns a valid action of shape (3,) in the correct bounds.
        """
        action = np.zeros(3, dtype=np.float32)
        
        # Extract info with defaults
        electricity_price = info.get('electricity_price', 0.25)
        soc = info.get('soc', 0.5)
        current_temp = obs[1] if len(obs) > 1 else 22.0  # obs[1] = zone temperature
        
        is_peak = electricity_price > 0.15  # Threshold for peak pricing
        
        # HVAC control (dim 0 and 1)
        if current_temp < 21.0:
            action[0] = 0.9  # Heat more (close to max 23.25)
            action[1] = 0.1  # Cool less
        elif current_temp > 27.0:
            action[0] = 0.1  # Heat less
            action[1] = 0.9  # Cool more (close to max 30)
        else:
            # Comfort zone → neutral setpoints
            action[0] = 0.5
            action[1] = 0.5
        
        # Battery control (dim 2: -1 to +1)
        if not is_peak and soc < 0.8:
            action[2] = 1.0   # Charge
        elif is_peak and soc > 0.2:
            action[2] = -1.0  # Discharge
        else:
            action[2] = 0.0   # Idle
        
        # Final safety clamp
        action = np.clip(action, self.low, self.high)
        
        return action


def get_mixed_policy_action(policy_type, obs, info, action_space, rule_policy, expert_agent=None):
    """
    Get action from mixed policy.
    
    Priorité :
    - 'random' → random
    - 'rule' → rule-based
    - 'expert' → expert if available, else fallback to rule-based (NOT random)
    """
    if policy_type == 'random':
        return action_space.sample()
    
    if policy_type == 'rule':
        return rule_policy.get_action(obs, info)
    
    if policy_type == 'expert':
        if expert_agent is not None:
            return expert_agent.select_action(obs, deterministic=True)
        else:
            # ⭐ Fallback sur rule-based au lieu de random
            return rule_policy.get_action(obs, info)
    
    # Fallback ultime (ne devrait jamais arriver)
    return action_space.sample()


def get_policy_distribution(num_samples, mix_ratios=None):
    """
    Generate policy types for dataset collection.
    
    Default: 80% random + 20% rule (expert temporarily disabled)
    """
    if mix_ratios is None:
        mix_ratios = {'random': 0.8, 'rule': 0.2}  # expert désactivé
    
    policies = []
    for policy_type, ratio in mix_ratios.items():
        count = int(num_samples * ratio)
        policies.extend([policy_type] * count)
    
    # Ajuster si léger déséquilibre dû à l'arrondi
    if len(policies) < num_samples:
        policies.extend(['random'] * (num_samples - len(policies)))
    
    np.random.shuffle(policies)
    return policies[:num_samples]