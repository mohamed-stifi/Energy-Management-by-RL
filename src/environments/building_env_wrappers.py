"""
Custom Gym Wrapper for Energy Management with Battery and Dynamic Pricing.

Extends Sinergym's EplusEnv to add:
- Battery storage (SOC, charge/discharge)
- Dynamic electricity pricing
- Solar PV production estimation
- Multi-objective reward (cost + comfort + battery cycling penalty)
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional
import sinergym


class BuildingBatteryEnv(gym.Wrapper):
    """
    Wrapper around Sinergym environment to add battery management.
    
    Battery Model:
    - Capacity: 10 kWh (configurable)
    - Efficiency: 90% (charge/discharge)
    - Power limits: ±5 kW
    - SOC bounds: 20% - 90% (to preserve battery life)
    
    Pricing Model:
    - On-peak: 7h-22h → 0.25 €/kWh
    - Off-peak: 22h-7h → 0.10 €/kWh
    
    Solar PV:
    - Max power: 5 kWp
    - Efficiency: 20%
    - Based on solar irradiance from weather
    
    Reward:
    - r = -cost - λ_comfort * comfort_penalty - λ_battery * cycling_penalty
    """
    
    def __init__(
        self,
        env: gym.Env,
        battery_capacity: float = 10.0,  # kWh
        battery_efficiency: float = 0.9,
        battery_power_max: float = 5.0,  # kW
        soc_min: float = 0.2,  # 20%
        soc_max: float = 0.9,  # 90%
        pv_max_power: float = 5.0,  # kWp
        pv_efficiency: float = 0.2,
        price_on_peak: float = 0.25,  # €/kWh
        price_off_peak: float = 0.10,  # €/kWh
        comfort_temp_range: Tuple[float, float] = (21.0, 25.0),  # °C
        lambda_comfort: float = 1.0,
        lambda_battery: float = 0.1,
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: Base Sinergym environment
            battery_capacity: Battery capacity in kWh
            battery_efficiency: Round-trip efficiency (0-1)
            battery_power_max: Max charge/discharge power in kW
            soc_min: Minimum State of Charge (0-1)
            soc_max: Maximum State of Charge (0-1)
            pv_max_power: Max PV power in kWp
            pv_efficiency: PV panel efficiency (0-1)
            price_on_peak: On-peak electricity price in €/kWh
            price_off_peak: Off-peak electricity price in €/kWh
            comfort_temp_range: (min, max) comfortable temperature in °C
            lambda_comfort: Weight for comfort penalty in reward
            lambda_battery: Weight for battery cycling penalty in reward
        """
        super().__init__(env)
        
        # Battery parameters
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.battery_power_max = battery_power_max
        self.soc_min = soc_min
        self.soc_max = soc_max
        
        # PV parameters
        self.pv_max_power = pv_max_power
        self.pv_efficiency = pv_efficiency
        
        # Pricing parameters
        self.price_on_peak = price_on_peak
        self.price_off_peak = price_off_peak
        
        # Comfort parameters
        self.comfort_temp_min, self.comfort_temp_max = comfort_temp_range
        
        # Reward weights
        self.lambda_comfort = lambda_comfort
        self.lambda_battery = lambda_battery
        
        # Battery state
        self.soc = 0.5  # Start at 50% SOC
        self.total_battery_cycles = 0.0
        
        # Modify observation space to include SOC
        original_obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.append(original_obs_space.low, 0.0),  # SOC min = 0%
            high=np.append(original_obs_space.high, 1.0),  # SOC max = 100%
            dtype=np.float32
        )
        
        # Modify action space to include battery control
        # Original action: HVAC setpoints
        # New action: HVAC setpoints + battery power (-1 to 1, scaled to ±battery_power_max)
        original_act_space = self.env.action_space
        self.action_space = gym.spaces.Box(
            low=np.append(original_act_space.low, -1.0),
            high=np.append(original_act_space.high, 1.0),
            dtype=np.float32
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and battery state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset battery to 50% SOC
        self.soc = 0.5
        self.total_battery_cycles = 0.0
        
        # Add SOC to observation
        obs_extended = np.append(obs, self.soc)
        
        return obs_extended, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: Array with [HVAC_actions..., battery_power_normalized]
        
        Returns:
            observation: Extended with SOC
            reward: Multi-objective reward
            terminated: Episode done flag
            truncated: Episode truncated flag
            info: Additional information
        """
        # Split action into HVAC and battery
        hvac_action = action[:-1]
        battery_power_normalized = action[-1]  # -1 to 1
        
        # Scale battery power to actual kW
        battery_power_kw = battery_power_normalized * self.battery_power_max
        
        # Step base environment with HVAC action
        obs, base_reward, terminated, truncated, info = self.env.step(hvac_action)
        
        # Extract current hour and temperature from observation
        # (Sinergym observations include time info - adapt indices based on your env)
        current_hour = self._extract_hour_from_info(info)
        current_temp = self._extract_temperature_from_obs(obs)
        solar_irradiance = self._extract_solar_irradiance_from_obs(obs)
        building_power = self._extract_building_power_from_obs(obs)
        
        # Calculate electricity price (dynamic)
        electricity_price = self._get_electricity_price(current_hour)
        
        # Calculate PV production
        pv_power = self._calculate_pv_production(solar_irradiance)
        
        # Update battery SOC
        soc_change, actual_battery_power = self._update_battery(battery_power_kw)
        
        # Calculate net grid power (positive = buying, negative = selling)
        net_grid_power = building_power + actual_battery_power - pv_power
        
        # Calculate cost (only if buying from grid)
        if net_grid_power > 0:
            electricity_cost = net_grid_power * electricity_price * (1/3600)  # kWh (timestep = 1h typically)
        else:
            # Selling to grid (feed-in tariff, typically lower)
            electricity_cost = net_grid_power * (electricity_price * 0.5) * (1/3600)
        
        # Calculate comfort penalty (quadratic outside comfort zone)
        comfort_penalty = self._calculate_comfort_penalty(current_temp)
        
        # Calculate battery cycling penalty (penalize frequent charge/discharge)
        cycling_penalty = abs(soc_change) / self.battery_capacity
        
        # Multi-objective reward
        reward = (
            -electricity_cost
            - self.lambda_comfort * comfort_penalty
            - self.lambda_battery * cycling_penalty
        )
        
        # Add SOC to observation
        obs_extended = np.append(obs, self.soc)
        
        # Add info for logging
        info['soc'] = self.soc
        info['battery_power'] = actual_battery_power
        info['pv_power'] = pv_power
        info['electricity_price'] = electricity_price
        info['electricity_cost'] = electricity_cost
        info['comfort_penalty'] = comfort_penalty
        info['cycling_penalty'] = cycling_penalty
        info['reward_components'] = {
            'cost': -electricity_cost,
            'comfort': -self.lambda_comfort * comfort_penalty,
            'battery': -self.lambda_battery * cycling_penalty
        }
        info['total_battery_cycles'] = self.total_battery_cycles
        info['net_grid_power'] = net_grid_power
        info['building_power'] = building_power
        
        
        return obs_extended, reward, terminated, truncated, info
    
    def _update_battery(self, power_kw: float) -> Tuple[float, float]:
        """
        Update battery SOC based on power command.
        
        Args:
            power_kw: Desired power (+ = charge, - = discharge)
        
        Returns:
            soc_change: Change in SOC (kWh)
            actual_power: Actual power after constraints (kW)
        """
        # Clip power to max limits
        power_kw = np.clip(power_kw, -self.battery_power_max, self.battery_power_max)
        
        # Calculate energy change (assuming 1-hour timestep, adjust if needed)
        timestep_hours = 1.0 / 3600.0  # Typically 900s = 0.25h in Sinergym
        energy_change_kwh = power_kw * timestep_hours
        
        # Apply efficiency
        if energy_change_kwh > 0:  # Charging
            energy_change_kwh *= self.battery_efficiency
        else:  # Discharging
            energy_change_kwh /= self.battery_efficiency
        
        # Calculate new SOC
        new_soc_kwh = self.soc * self.battery_capacity + energy_change_kwh
        
        # Clip to SOC bounds
        new_soc_kwh = np.clip(
            new_soc_kwh,
            self.soc_min * self.battery_capacity,
            self.soc_max * self.battery_capacity
        )
        
        # Actual energy change after clipping
        actual_energy_change = new_soc_kwh - self.soc * self.battery_capacity
        
        # Update SOC
        self.soc = new_soc_kwh / self.battery_capacity
        
        # Track total cycles
        self.total_battery_cycles += abs(actual_energy_change) / (2 * self.battery_capacity)
        
        # Actual power (reverse calculate from actual energy change)
        actual_power = actual_energy_change / timestep_hours
        
        return actual_energy_change, actual_power
    
    def _get_electricity_price(self, hour: int) -> float:
        """Get electricity price based on time of day."""
        if 7 <= hour < 22:  # On-peak: 7h-22h
            return self.price_on_peak
        else:  # Off-peak: 22h-7h
            return self.price_off_peak
    
    def _calculate_pv_production(self, solar_irradiance: float) -> float:
        """
        Calculate PV power production.
        
        Args:
            solar_irradiance: Solar irradiance in W/m²
        
        Returns:
            pv_power: PV production in kW
        """
        # Simple model: P = irradiance * area * efficiency
        # Assuming 1 kWp = 6.5 m² of panels
        panel_area = self.pv_max_power * 6.5  # m²
        pv_power_w = solar_irradiance * panel_area * self.pv_efficiency
        pv_power_kw = pv_power_w / 1000.0
        
        # Clip to max power
        return min(pv_power_kw, self.pv_max_power)
    
    def _calculate_comfort_penalty(self, temp: float) -> float:
        """
        Calculate comfort penalty (quadratic outside comfort zone).
        
        Args:
            temp: Current indoor temperature in °C
        
        Returns:
            penalty: Positive value (0 if in comfort zone)
        """
        if temp < self.comfort_temp_min:
            return (self.comfort_temp_min - temp) ** 2
        elif temp > self.comfort_temp_max:
            return (temp - self.comfort_temp_max) ** 2
        else:
            return 0.0
    
    
    def _extract_hour_from_info(self, info: Dict) -> int:
        """Extract current hour from info dictionary."""
        return int(info.get('hour', 0))
    
    
    def _extract_temperature_from_obs(self, obs: np.ndarray) -> float:
        """
        Extract indoor temperature from observation.
        Based on diagnostic: obs[9] seems to be temperature (≈19.95)
        """
        try:
            return float(obs[9])
        except (IndexError, ValueError):
            return 22.0  # Safe default
    
    
    def _extract_solar_irradiance_from_obs(self, obs: np.ndarray) -> float:
        """
        Extract solar irradiance from observation (W/m²).
        Based on diagnostic: obs[11] appears to be irradiance (0.0 at night)
        """
        try:
            return float(obs[11])
        except (IndexError, ValueError):
            return 0.0
    
    
    def _extract_building_power_from_obs(self, obs: np.ndarray) -> float:
        """
        Extract building power consumption from observation (W).
        Based on diagnostic: obs[16] is large (≈106152) → likely power in W
        Convert to kW by dividing by 1000.
        """
        try:
            power_w = float(obs[16])
            return power_w / 1000.0  # Convert W to kW
        except (IndexError, ValueError):
            return 0.0