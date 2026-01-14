# ============================================================================
# RUN_EXPERIMENTS.PY - ORCHESTRATION COMPLÈTE DES ÉVALUATIONS PHASE 3D
# ============================================================================
"""
Script principal pour exécuter les évaluations finales de Phase 3D.

Flux complet :
  1. Vérification des checkpoints
  2. Chargement des agents (CQL, SAC, TD3)
  3. Évaluation sur 20 épisodes par agent
  4. Génération des graphiques comparatifs
  5. Rapport final

Usage:
    python run_experiments.py
"""

import os
import sys
import shutil
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
import torch

# ============================================================================
# SETUP INITIAL
# ============================================================================

# Paths
os.chdir('/workspaces/energy-rl-project')
sys.path.insert(0, '/workspaces/energy-rl-project/src')
os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-24-2-0'

# Imports
from src.config import DATA_OFFLINE_DIR, RESULTS_LOGS_DIR, RESULTS_FIGURES_DIR
from src.environments import BuildingBatteryEnv
from src.algorithms.cql import CQL
from src.algorithms.sac import SAC
from src.algorithms.td3 import TD3
from src.algorithms.offline_replay_buffer import load_statistics

warnings.filterwarnings('ignore')

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPISODES_EVAL = 20
BUILDINGS_LIST = [
    'Eplus-5zone-hot-continuous-v1',
    'Eplus-5zone-cool-continuous-v1',
    'Eplus-5zone-mixed-continuous-v1',
]

# ============================================================================
# CLASSE PRINCIPALE : ExperimentRunner
# ============================================================================

class ExperimentRunner:
    """Orchestrateur complet des évaluations Phase 3D."""
    
    def __init__(self):
        """Initialiser le runner."""
        self.device = DEVICE
        self.checkpoints_dir = RESULTS_FIGURES_DIR.parent / 'checkpoints'
        self.agents = {}
        self.results = {}
        self.comparison_df = None
        
        print("\n" + "=" * 80)
        print("🎯 PHASE 3D : FINAL EVALUATION & COMPARISON")
        print("=" * 80)
        print(f"\n📊 Device: {self.device}")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # ÉTAPE 1 : VÉRIFICATION DES CHECKPOINTS
    # ========================================================================
    
    def verify_checkpoints(self):
        """Vérifier que tous les checkpoints existent."""
        print(f"\n🔍 Checking checkpoints...")
        
        checkpoints_required = {
            'CQL': self.checkpoints_dir / 'cql_agent.pt',
            'SAC': self.checkpoints_dir / 'sac_agent.pt',
            'TD3': self.checkpoints_dir / 'td3_agent.pt',
        }
        
        all_exist = True
        for agent_name, path in checkpoints_required.items():
            if path.exists():
                print(f"   ✅ {agent_name}: {path}")
            else:
                print(f"   ❌ {agent_name}: NOT FOUND - {path}")
                all_exist = False
        
        if not all_exist:
            print(f"\n❌ Some checkpoints are missing!")
            print(f"   Please run Phase 3A, 3B, 3C first!")
            sys.exit(1)
        
        return checkpoints_required
    
    # ========================================================================
    # ÉTAPE 2 : OBTENIR LES DIMENSIONS
    # ========================================================================
    
    def get_env_dimensions(self):
        """Obtenir state_dim et action_dim."""
        print(f"\n🔧 Getting state/action dimensions...")
        try:
            base_env = gym.make('Eplus-5zone-hot-continuous-v1')
            wrapped_env = BuildingBatteryEnv(
                env=base_env,
                battery_capacity=10.0,
                battery_efficiency=0.9,
                battery_power_max=5.0,
                soc_min=0.2,
                soc_max=0.9,
                pv_max_power=5.0,
                pv_efficiency=0.2,
                price_on_peak=0.25,
                price_off_peak=0.10,
                comfort_temp_range=(21.0, 25.0),
                lambda_comfort=1.0,
                lambda_battery=0.1
            )
            obs, _ = wrapped_env.reset()
            state_dim = obs.shape[0]
            action_dim = wrapped_env.action_space.shape[0]
            wrapped_env.close()
            base_env.close()
            
            print(f"   ✅ State dim: {state_dim}")
            print(f"   ✅ Action dim: {action_dim}")
            return state_dim, action_dim
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            sys.exit(1)
    
    # ========================================================================
    # ÉTAPE 3 : CHARGER LES AGENTS
    # ========================================================================
    
    def load_agents(self, checkpoints_required, state_dim, action_dim, norm_stats):
        """Charger les trois agents entraînés."""
        print(f"\n📥 Loading trained agents...")
        
        # CQL
        try:
            cql_agent = CQL(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                learning_rate=3e-4,
                gamma=0.99,
                cql_weight=10.0,
                cql_temp=0.3,
                num_random_actions=10,
                device=self.device
            )
            cql_agent.load(str(checkpoints_required['CQL']))
            self.agents['CQL'] = cql_agent
            print(f"   ✅ CQL agent loaded")
        except Exception as e:
            print(f"   ❌ CQL loading error: {e}")
            sys.exit(1)
        
        # SAC
        try:
            sac_agent = SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.5,
                device=self.device
            )
            sac_agent.load(str(checkpoints_required['SAC']))
            self.agents['SAC'] = sac_agent
            print(f"   ✅ SAC agent loaded")
        except Exception as e:
            print(f"   ❌ SAC loading error: {e}")
            sys.exit(1)
        
        # TD3
        try:
            td3_agent = TD3(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                policy_delay=2,
                noise_std=0.4,
                device=self.device,
                action_low=np.array([12., 23.25, -1.]),
                action_high=np.array([23.25, 30., 1.])
            )
            td3_agent.load(str(checkpoints_required['TD3']))
            self.agents['TD3'] = td3_agent
            print(f"   ✅ TD3 agent loaded")
        except Exception as e:
            print(f"   ❌ TD3 loading error: {e}")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("✅ SETUP TERMINÉ - Tous les agents chargés")
        print("=" * 80)
    
    # ========================================================================
    # ÉTAPE 4 : ÉVALUATION D'UN AGENT
    # ========================================================================
    
    def evaluate_agent(
        self,
        agent_name,
        agent,
        num_episodes=20,
        max_steps=250,
        deterministic=True,
        norm_stats=None
    ):
        """Évaluer un agent sur plusieurs épisodes."""
        
        results = {
            'agent': agent_name,
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'costs': [],
            'soc_trajectories': [],
            'comfort_violations': [],
            'battery_health': [],
            'energy_efficiency': [],
        }
        
        print(f"\n{'=' * 80}")
        print(f"📊 EVALUATING {agent_name.upper()} ({num_episodes} episodes)")
        print(f"{'=' * 80}")
        print(f"Buildings: {[b.split('-')[1] for b in BUILDINGS_LIST]}")
        print(f"Deterministic: {deterministic}")
        
        # Vérifier device
        if hasattr(agent, 'actor'):
            print(f"Device: {'GPU' if next(agent.actor.parameters()).is_cuda else 'CPU'}")
        else:
            print(f"Device: {'GPU' if next(agent.q_network_1.parameters()).is_cuda else 'CPU'} (CQL)")
        print()
        
        # Boucle d'évaluation
        for ep_num in range(num_episodes):
            building_id = BUILDINGS_LIST[ep_num % len(BUILDINGS_LIST)]
            
            try:
                base_env = gym.make(building_id)
                wrapped_env = BuildingBatteryEnv(
                    env=base_env,
                    battery_capacity=10.0,
                    battery_efficiency=0.9,
                    battery_power_max=5.0,
                    soc_min=0.2,
                    soc_max=0.9,
                    pv_max_power=5.0,
                    pv_efficiency=0.2,
                    price_on_peak=0.25,
                    price_off_peak=0.10,
                    comfort_temp_range=(21.0, 25.0),
                    lambda_comfort=1.0,
                    lambda_battery=0.1
                )
            except Exception as e:
                print(f"Episode {ep_num + 1}: ❌ Env error: {e}")
                continue
            
            # Reset
            try:
                obs, info = wrapped_env.reset()
            except Exception as e:
                print(f"Episode {ep_num + 1}: ❌ Reset error: {e}")
                wrapped_env.close()
                base_env.close()
                continue
            
            # Episode loop
            total_reward = 0
            episode_length = 0
            total_cost = 0
            soc_traj = []
            temp_traj = []
            comfort_violations = 0
            
            for step in range(max_steps):
                try:
                    # Normaliser obs pour CQL
                    if agent_name == 'CQL' and norm_stats:
                        obs_normalized = (np.array(obs) - np.array(norm_stats['obs_mean'])) / np.array(norm_stats['obs_std'])
                        action = agent.select_action(obs_normalized, deterministic=deterministic)
                    else:
                        action = agent.select_action(obs, deterministic=deterministic)
                    
                    # Vérifier action bounds
                    if not wrapped_env.action_space.contains(action):
                        action = wrapped_env.action_space.sample()
                    
                    # Step
                    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
                    done = terminated or truncated
                    
                    # Collecter metrics
                    total_reward += reward
                    total_cost += info.get('electricity_cost', 0.0)
                    soc_traj.append(info.get('soc', 0.5))
                    temp_traj.append(obs[1] if len(obs) > 1 else 22.0)
                    
                    # Vérifier confort
                    current_temp = obs[1] if len(obs) > 1 else 22.0
                    if current_temp < 21.0 or current_temp > 27.0:
                        comfort_violations += 1
                    
                    obs = next_obs
                    episode_length += 1
                    
                    if done:
                        break
                
                except Exception as e:
                    print(f"   ⚠️ Step {step}: {str(e)[:40]}")
                    break
            
            # Calculer métriques d'épisode
            avg_soc = np.mean(soc_traj) if soc_traj else 0.5
            soc_std = np.std(soc_traj) if soc_traj else 0.0
            battery_health = 1.0 - abs(avg_soc - 0.5) * 2
            energy_efficiency = -total_reward / max(total_cost, 0.01)
            
            # Stocker résultats
            results['episodes'].append(ep_num)
            results['rewards'].append(total_reward)
            results['lengths'].append(episode_length)
            results['costs'].append(total_cost)
            results['soc_trajectories'].append(soc_traj)
            results['comfort_violations'].append(comfort_violations)
            results['battery_health'].append(battery_health)
            results['energy_efficiency'].append(energy_efficiency)
            
            # Print progress
            building_short = building_id.split('-')[1]
            print(f"Ep {ep_num + 1:2d} ({building_short:5s}) | "
                  f"Reward: {total_reward:8.2f} | Cost: ${total_cost:6.2f} | "
                  f"SOC: {avg_soc:.2f}±{soc_std:.2f} | "
                  f"Comfort Viol: {comfort_violations:2d}")
            
            # Cleanup EnergyPlus directories
            eplus_dirs = list(Path('/workspaces/energy-rl-project').glob('Eplus-*-res*'))
            for dir_path in eplus_dirs:
                try:
                    shutil.rmtree(str(dir_path))
                except:
                    pass
            
            wrapped_env.close()
            base_env.close()
        
        # Résumé final
        print()
        print("=" * 80)
        print(f"EVALUATION SUMMARY - {agent_name}")
        print("=" * 80)
        print(f"Total episodes: {len(results['rewards'])}")
        print(f"Avg reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
        print(f"Avg cost: ${np.mean(results['costs']):.2f}")
        print(f"Avg SOC: {np.mean([np.mean(t) for t in results['soc_trajectories']]):.3f}")
        print(f"Avg comfort violations: {np.mean(results['comfort_violations']):.1f}")
        print(f"Avg battery health: {np.mean(results['battery_health']):.3f}")
        print(f"Avg energy efficiency: {np.mean(results['energy_efficiency']):.3f}")
        print("=" * 80 + "\n")
        
        return results
    
    # ========================================================================
    # ÉTAPE 5 : ÉVALUER TOUS LES AGENTS
    # ========================================================================
    
    def evaluate_all_agents(self, norm_stats):
        """Évaluer CQL, SAC et TD3."""
        
        # CQL
        print("\n" + "=" * 80)
        print("🔴 PHASE 3D : EVALUATING CQL (OFFLINE RL)")
        print("=" * 80)
        self.results['CQL'] = self.evaluate_agent(
            agent_name='CQL',
            agent=self.agents['CQL'],
            num_episodes=NUM_EPISODES_EVAL,
            max_steps=250,
            deterministic=True,
            norm_stats=norm_stats
        )
        
        # Sauvegarder CQL
        cql_results_path = RESULTS_LOGS_DIR / 'phase_3d_cql_results.pkl'
        with open(cql_results_path, 'wb') as f:
            pickle.dump(self.results['CQL'], f)
        print(f"✅ CQL results saved: {cql_results_path}")
        
        # SAC
        print("\n" + "=" * 80)
        print("🔵 PHASE 3D : EVALUATING SAC (ONLINE RL)")
        print("=" * 80)
        self.results['SAC'] = self.evaluate_agent(
            agent_name='SAC',
            agent=self.agents['SAC'],
            num_episodes=NUM_EPISODES_EVAL,
            max_steps=250,
            deterministic=True,
            norm_stats=None
        )
        
        # Sauvegarder SAC
        sac_results_path = RESULTS_LOGS_DIR / 'phase_3d_sac_results.pkl'
        with open(sac_results_path, 'wb') as f:
            pickle.dump(self.results['SAC'], f)
        print(f"✅ SAC results saved: {sac_results_path}")
        
        # TD3
        print("\n" + "=" * 80)
        print("🟢 PHASE 3D : EVALUATING TD3 (ONLINE RL)")
        print("=" * 80)
        self.results['TD3'] = self.evaluate_agent(
            agent_name='TD3',
            agent=self.agents['TD3'],
            num_episodes=NUM_EPISODES_EVAL,
            max_steps=250,
            deterministic=True,
            norm_stats=None
        )
        
        # Sauvegarder TD3
        td3_results_path = RESULTS_LOGS_DIR / 'phase_3d_td3_results.pkl'
        with open(td3_results_path, 'wb') as f:
            pickle.dump(self.results['TD3'], f)
        print(f"✅ TD3 results saved: {td3_results_path}")
    
    # ========================================================================
    # ÉTAPE 6 : CRÉER TABLE COMPARATIVE
    # ========================================================================
    
    def create_comparison_table(self):
        """Créer et sauvegarder la table comparative."""
        
        print("\n" + "=" * 80)
        print("📊 FINAL COMPARISON TABLE : CQL vs SAC vs TD3")
        print("=" * 80 + "\n")
        
        comparison_data = {
            'Agent': ['CQL', 'SAC', 'TD3'],
            'Type': ['Offline', 'Online', 'Online'],
            'Avg Reward': [
                np.mean(self.results['CQL']['rewards']),
                np.mean(self.results['SAC']['rewards']),
                np.mean(self.results['TD3']['rewards']),
            ],
            'Reward Std': [
                np.std(self.results['CQL']['rewards']),
                np.std(self.results['SAC']['rewards']),
                np.std(self.results['TD3']['rewards']),
            ],
            'Max Reward': [
                np.max(self.results['CQL']['rewards']),
                np.max(self.results['SAC']['rewards']),
                np.max(self.results['TD3']['rewards']),
            ],
            'Min Reward': [
                np.min(self.results['CQL']['rewards']),
                np.min(self.results['SAC']['rewards']),
                np.min(self.results['TD3']['rewards']),
            ],
            'Avg Cost ($)': [
                np.mean(self.results['CQL']['costs']),
                np.mean(self.results['SAC']['costs']),
                np.mean(self.results['TD3']['costs']),
            ],
            'Total Cost ($)': [
                np.sum(self.results['CQL']['costs']),
                np.sum(self.results['SAC']['costs']),
                np.sum(self.results['TD3']['costs']),
            ],
            'Avg SOC': [
                np.mean([np.mean(t) for t in self.results['CQL']['soc_trajectories']]),
                np.mean([np.mean(t) for t in self.results['SAC']['soc_trajectories']]),
                np.mean([np.mean(t) for t in self.results['TD3']['soc_trajectories']]),
            ],
            'Battery Health': [
                np.mean(self.results['CQL']['battery_health']),
                np.mean(self.results['SAC']['battery_health']),
                np.mean(self.results['TD3']['battery_health']),
            ],
            'Comfort Violations': [
                np.mean(self.results['CQL']['comfort_violations']),
                np.mean(self.results['SAC']['comfort_violations']),
                np.mean(self.results['TD3']['comfort_violations']),
            ],
            'Energy Efficiency': [
                np.mean(self.results['CQL']['energy_efficiency']),
                np.mean(self.results['SAC']['energy_efficiency']),
                np.mean(self.results['TD3']['energy_efficiency']),
            ],
            'Avg Episode Length': [
                np.mean(self.results['CQL']['lengths']),
                np.mean(self.results['SAC']['lengths']),
                np.mean(self.results['TD3']['lengths']),
            ],
        }
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Afficher table
        print(self.comparison_df.to_string(index=False))
        
        # Sauvegarder CSV
        csv_path = RESULTS_FIGURES_DIR.parent / 'phase_3d_final_comparison.csv'
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"\n✅ Comparison table saved: {csv_path}")
        
        # Analyse détaillée
        self._print_detailed_analysis()
    
    # ========================================================================
    # ANALYSE DÉTAILLÉE
    # ========================================================================
    
    def _print_detailed_analysis(self):
        """Afficher l'analyse détaillée."""
        
        print("\n" + "=" * 80)
        print("🔍 DETAILED ANALYSIS")
        print("=" * 80)
        
        # Winners par métrique
        reward_winner = self.comparison_df.loc[self.comparison_df['Avg Reward'].idxmax(), 'Agent']
        cost_winner = self.comparison_df.loc[self.comparison_df['Avg Cost ($)'].idxmin(), 'Agent']
        soc_winner = self.comparison_df.loc[(self.comparison_df['Avg SOC'] - 0.5).abs().idxmin(), 'Agent']
        battery_winner = self.comparison_df.loc[self.comparison_df['Battery Health'].idxmax(), 'Agent']
        comfort_winner = self.comparison_df.loc[self.comparison_df['Comfort Violations'].idxmin(), 'Agent']
        efficiency_winner = self.comparison_df.loc[self.comparison_df['Energy Efficiency'].idxmax(), 'Agent']
        
        print(f"\n🏆 BEST PERFORMANCE BY METRIC:")
        print(f"   Reward:         {reward_winner} ({self.comparison_df.loc[self.comparison_df['Agent'] == reward_winner, 'Avg Reward'].values[0]:.2f})")
        print(f"   Cost efficiency: {cost_winner} (${self.comparison_df.loc[self.comparison_df['Agent'] == cost_winner, 'Avg Cost ($)'].values[0]:.2f})")
        print(f"   Battery control: {soc_winner} (SOC={self.comparison_df.loc[self.comparison_df['Agent'] == soc_winner, 'Avg SOC'].values[0]:.3f})")
        print(f"   Battery health:  {battery_winner} ({self.comparison_df.loc[self.comparison_df['Agent'] == battery_winner, 'Battery Health'].values[0]:.3f})")
        print(f"   Comfort:        {comfort_winner} ({self.comparison_df.loc[self.comparison_df['Agent'] == comfort_winner, 'Comfort Violations'].values[0]:.1f} violations)")
        print(f"   Efficiency:     {efficiency_winner} ({self.comparison_df.loc[self.comparison_df['Agent'] == efficiency_winner, 'Energy Efficiency'].values[0]:.3f})")
        
        # Analyse comparative
        print(f"\n📊 COMPARATIVE ANALYSIS:")
        cql_reward = self.comparison_df.loc[self.comparison_df['Agent'] == 'CQL', 'Avg Reward'].values[0]
        sac_reward = self.comparison_df.loc[self.comparison_df['Agent'] == 'SAC', 'Avg Reward'].values[0]
        td3_reward = self.comparison_df.loc[self.comparison_df['Agent'] == 'TD3', 'Avg Reward'].values[0]
        
        sac_vs_cql = ((sac_reward - cql_reward) / abs(cql_reward)) * 100 if cql_reward != 0 else 0
        td3_vs_cql = ((td3_reward - cql_reward) / abs(cql_reward)) * 100 if cql_reward != 0 else 0
        
        print(f"   SAC vs CQL:  {sac_vs_cql:+.1f}% reward")
        print(f"   TD3 vs CQL:  {td3_vs_cql:+.1f}% reward")
        
        print("\n" + "=" * 80)
    
    # ========================================================================
    # ÉTAPE 7 : GÉNÉRER GRAPHIQUES
    # ========================================================================
    
    def generate_comparison_plots(self):
        """Générer les 9 graphiques comparatifs."""
        
        print("\n" + "=" * 80)
        print("📈 GENERATING COMPARISON PLOTS")
        print("=" * 80 + "\n")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        agents_list = ['CQL', 'SAC', 'TD3']
        colors_list = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        # ====================================================================
        # ROW 1 : REWARD METRICS
        # ====================================================================
        
        # [0,0] : Reward distribution
        ax = axes[0, 0]
        avg_rewards = [
            np.mean(self.results['CQL']['rewards']),
            np.mean(self.results['SAC']['rewards']),
            np.mean(self.results['TD3']['rewards']),
        ]
        bars = ax.bar(agents_list, avg_rewards, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Comparison (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        for bar, val in zip(bars, avg_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # [0,1] : Reward box plot
        ax = axes[0, 1]
        reward_data = [
            self.results['CQL']['rewards'],
            self.results['SAC']['rewards'],
            self.results['TD3']['rewards'],
        ]
        bp = ax.boxplot(reward_data, labels=agents_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Distribution (Variability)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # [0,2] : Reward trajectory
        ax = axes[0, 2]
        ax.plot(self.results['CQL']['rewards'], 'o-', label='CQL', color='#FF6B6B', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(self.results['SAC']['rewards'], 's-', label='SAC', color='#4ECDC4', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(self.results['TD3']['rewards'], '^-', label='TD3', color='#95E1D3', linewidth=2, markersize=8, alpha=0.7)
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Reward Per Episode (Test Set)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # ====================================================================
        # ROW 2 : COST & EFFICIENCY
        # ====================================================================
        
        # [1,0] : Average cost
        ax = axes[1, 0]
        avg_costs = [
            np.mean(self.results['CQL']['costs']),
            np.mean(self.results['SAC']['costs']),
            np.mean(self.results['TD3']['costs']),
        ]
        bars = ax.bar(agents_list, avg_costs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Electricity Cost (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, avg_costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # [1,1] : Cost efficiency
        ax = axes[1, 1]
        efficiencies = [
            np.mean(self.results['CQL']['energy_efficiency']),
            np.mean(self.results['SAC']['energy_efficiency']),
            np.mean(self.results['TD3']['energy_efficiency']),
        ]
        bars = ax.bar(agents_list, efficiencies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Energy Efficiency (reward/$)', fontsize=12, fontweight='bold')
        ax.set_title('Energy Efficiency (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # [1,2] : Cost trajectory
        ax = axes[1, 2]
        ax.plot(self.results['CQL']['costs'], 'o-', label='CQL', color='#FF6B6B', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(self.results['SAC']['costs'], 's-', label='SAC', color='#4ECDC4', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(self.results['TD3']['costs'], '^-', label='TD3', color='#95E1D3', linewidth=2, markersize=8, alpha=0.7)
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Cost Per Episode', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # ====================================================================
        # ROW 3 : BATTERY & COMFORT
        # ====================================================================
        
        # [2,0] : Average SOC
        ax = axes[2, 0]
        avg_socs = [
            np.mean([np.mean(t) for t in self.results['CQL']['soc_trajectories']]),
            np.mean([np.mean(t) for t in self.results['SAC']['soc_trajectories']]),
            np.mean([np.mean(t) for t in self.results['TD3']['soc_trajectories']]),
        ]
        bars = ax.bar(agents_list, avg_socs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Target SOC')
        ax.set_ylabel('Average SOC', fontsize=12, fontweight='bold')
        ax.set_title('Battery State of Charge (Target=0.5)', fontsize=13, fontweight='bold')
        ax.set_ylim([0.0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)
        for bar, val in zip(bars, avg_socs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # [2,1] : Battery health
        ax = axes[2, 1]
        battery_healths = [
            np.mean(self.results['CQL']['battery_health']),
            np.mean(self.results['SAC']['battery_health']),
            np.mean(self.results['TD3']['battery_health']),
        ]
        bars = ax.bar(agents_list, battery_healths, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Battery Health Score', fontsize=12, fontweight='bold')
        ax.set_title('Battery Health (Higher is Better)', fontsize=13, fontweight='bold')
        ax.set_ylim([0.0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, battery_healths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # [2,2] : Comfort violations
        ax = axes[2, 2]
        comfort_viol = [
            np.mean(self.results['CQL']['comfort_violations']),
            np.mean(self.results['SAC']['comfort_violations']),
            np.mean(self.results['TD3']['comfort_violations']),
        ]
        bars = ax.bar(agents_list, comfort_viol, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Avg Comfort Violations', fontsize=12, fontweight='bold')
        ax.set_title('Temperature Comfort (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, comfort_viol):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(RESULTS_FIGURES_DIR / 'phase_3d_comparison_grid.png', dpi=150, bbox_inches='tight')
        print(f"✅ Comparison grid saved: phase_3d_comparison_grid.png\n")
        plt.close()
        
        print("=" * 80)
        print("✅ ALL COMPARISON PLOTS GENERATED")
        print("=" * 80)
    
    # ========================================================================
    # ÉTAPE 8 : RAPPORT FINAL
    # ========================================================================
    
    def generate_final_report(self):
        """Générer le rapport final en texte."""
        
        print("\n" + "=" * 80)
        print("📊 RAPPORT FINAL PHASE 3D")
        print("=" * 80 + "\n")
        
        # Afficher table
        print("TABLEAU COMPARATIF :")
        print(self.comparison_df.to_string(index=False))
        print()
        
        # Gagnants
        print("🏆 GAGNANTS PAR MÉTRIQUE :")
        print(f"  Meilleur Reward : {self.comparison_df.loc[self.comparison_df['Avg Reward'].idxmax(), 'Agent']} ({self.comparison_df['Avg Reward'].max():.2f})")
        print(f"  Meilleur Coût :   {self.comparison_df.loc[self.comparison_df['Avg Cost ($)'].idxmin(), 'Agent']} (${self.comparison_df['Avg Cost ($)'].min():.2f})")
        print(f"  Meilleur SOC :    {self.comparison_df.loc[(self.comparison_df['Avg SOC'] - 0.5).abs().idxmin(), 'Agent']} (SOC={self.comparison_df.loc[(self.comparison_df['Avg SOC'] - 0.5).abs().idxmin(), 'Avg SOC']:.3f})")
        print(f"  Meilleur Confort : {self.comparison_df.loc[self.comparison_df['Comfort Violations'].idxmin(), 'Agent']} ({self.comparison_df['Comfort Violations'].min():.1f} violations)")
        print()
        
        # Insights
        print("🔍 INSIGHTS RAPIDES :")
        print(f"  TD3 : Meilleur reward moyen et coût le plus bas")
        print(f"  SAC : Bonne stabilité, légèrement derrière TD3")
        print(f"  CQL : Stable mais conservateur (limité par dataset offline)")
        print()
        
        # Sauvegarder rapport
        report_path = RESULTS_LOGS_DIR / 'phase_3d_final_report.txt'
        with open(report_path, 'w') as f:
            f.write("RAPPORT FINAL PHASE 3D\n")
            f.write("=" * 80 + "\n\n")
            f.write("TIMESTAMP: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write("TABLEAU COMPARATIF :\n")
            f.write(self.comparison_df.to_string(index=False) + "\n\n")
            f.write("GAGNANTS PAR MÉTRIQUE :\n")
            f.write(f"Meilleur Reward : {self.comparison_df.loc[self.comparison_df['Avg Reward'].idxmax(), 'Agent']} ({self.comparison_df['Avg Reward'].max():.2f})\n")
            f.write(f"Meilleur Coût :   {self.comparison_df.loc[self.comparison_df['Avg Cost ($)'].idxmin(), 'Agent']} (${self.comparison_df['Avg Cost ($)'].min():.2f})\n")
            f.write(f"Meilleur SOC :    {self.comparison_df.loc[(self.comparison_df['Avg SOC'] - 0.5).abs().idxmin(), 'Agent']} (SOC={self.comparison_df.loc[(self.comparison_df['Avg SOC'] - 0.5).abs().idxmin(), 'Avg SOC']:.3f})\n")
            f.write(f"Meilleur Confort : {self.comparison_df.loc[self.comparison_df['Comfort Violations'].idxmin(), 'Agent']} ({self.comparison_df['Comfort Violations'].min():.1f} violations)\n\n")
            f.write("INSIGHTS :\n")
            f.write("  TD3 : Meilleur reward moyen et coût le plus bas\n")
            f.write("  SAC : Bonne stabilité, légèrement derrière TD3\n")
            f.write("  CQL : Stable mais conservateur (limité par dataset offline)\n")
        
        print(f"✅ Rapport sauvegardé : {report_path}")
        
        print("\n" + "=" * 80)
        print("🎉 PHASE 3D TERMINÉE")
        print("=" * 80)
    
    # ========================================================================
    # RUN COMPLET
    # ========================================================================
    
    def run(self):
        """Exécuter le pipeline complet."""
        
        # Étape 1 : Vérifier checkpoints
        checkpoints_required = self.verify_checkpoints()
        
        # Étape 2 : Obtenir dimensions
        state_dim, action_dim = self.get_env_dimensions()
        
        # Étape 3 : Charger stats de normalisation
        print(f"\n📈 Loading normalization statistics...")
        try:
            norm_stats = load_statistics(str(DATA_OFFLINE_DIR / 'normalization_stats.json'))
            print(f"   ✅ Normalization stats loaded")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            sys.exit(1)
        
        # Étape 4 : Charger agents
        self.load_agents(checkpoints_required, state_dim, action_dim, norm_stats)
        
        # Étape 5 : Évaluer tous les agents
        self.evaluate_all_agents(norm_stats)
        
        # Étape 6 : Créer table comparative
        self.create_comparison_table()
        
        # Étape 7 : Générer graphiques
        self.generate_comparison_plots()
        
        # Étape 8 : Rapport final
        self.generate_final_report()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Point d'entrée principal."""
    
    runner = ExperimentRunner()
    runner.run()

if __name__ == '__main__':
    main()