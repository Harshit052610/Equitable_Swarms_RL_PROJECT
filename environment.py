import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv


class EquitableSwarmEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for Equitable-Swarm Disaster Relief.
    
    Grid: 10x10 toroidal world with 3 drones delivering aid to disaster zones.
    Uses Jain's Fairness Index to incentivize equitable resource distribution.
    """
    
    metadata = {
        "name": "equitable_swarm_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True
    }
    
    def __init__(self, render_mode: Optional[str] = None, random_seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        
        self.grid_size = 10
        self.num_drones = 3
        self.max_steps = 200
        self.max_demand = 9
        
        self.agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.possible_agents = self.agents.copy()
        
        self.action_space = Discrete(5)
        self.observation_space = Box(
            low=0,
            high=self.max_demand,
            shape=(2, 3, 3),
            dtype=np.int32
        )
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self._grid: Optional[np.ndarray] = None
        self._agent_positions: Dict[str, Tuple[int, int]] = {}
        self._cumulative_aid: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._trajectories: Dict[str, List[Tuple[int, int]]] = {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        
        self._grid = np.random.randint(0, self.max_demand + 1, size=(self.grid_size, self.grid_size), dtype=np.int32)
        self._cumulative_aid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._step_count = 0
        
        for agent in self.agents:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self._agent_positions[agent] = (x, y)
            self._trajectories[agent] = [(x, y)]
        
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        self._step_count += 1
        
        new_positions = {}
        for agent in self.agents:
            action = actions[agent]
            x, y = self._agent_positions[agent]
            
            if action == 1:
                x = (x - 1) % self.grid_size
            elif action == 2:
                x = (x + 1) % self.grid_size
            elif action == 3:
                y = (y + 1) % self.grid_size
            elif action == 4:
                y = (y - 1) % self.grid_size
            
            new_positions[agent] = (x, y)
        
        self._agent_positions = new_positions
        
        for agent in self.agents:
            x, y = self._agent_positions[agent]
            self._trajectories[agent].append((x, y))
        
        aid_delivered_this_step = 0
        for agent in self.agents:
            x, y = self._agent_positions[agent]
            if self._grid[x, y] > 0:
                aid_delivered = min(self._grid[x, y], 1)
                self._grid[x, y] -= aid_delivered
                self._cumulative_aid[x, y] += aid_delivered
                aid_delivered_this_step += aid_delivered
        
        jain_index = self._compute_jain_index()
        team_reward = aid_delivered_this_step * jain_index if jain_index > 0 else 0.0
        
        total_remaining_demand = np.sum(self._grid)
        terminated = total_remaining_demand == 0 or self._step_count >= self.max_steps
        truncated = False
        
        rewards = {agent: team_reward for agent in self.agents}
        dones = {agent: terminated for agent in self.agents}
        
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {
            agent: {
                "total_aid_delivered": np.sum(self._cumulative_aid),
                "jain_index": jain_index,
                "remaining_demand": total_remaining_demand
            } 
            for agent in self.agents
        }
        
        return observations, rewards, dones, dones, infos
    
    def _observe(self, agent: str) -> np.ndarray:
        x, y = self._agent_positions[agent]
        
        obs = np.zeros((2, 3, 3), dtype=np.int32)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = (x + dx) % self.grid_size
                ny = (y + dy) % self.grid_size
                
                obs[0, dx + 1, dy + 1] = self._grid[nx, ny]
                
                for other_agent in self.agents:
                    if other_agent != agent:
                        ox, oy = self._agent_positions[other_agent]
                        if nx == ox and ny == oy:
                            obs[1, dx + 1, dy + 1] = 1
        
        return obs
    
    def _compute_jain_index(self) -> float:
        total_aid = np.sum(self._cumulative_aid)
        if total_aid == 0:
            return 0.0
        
        cell_aid = self._cumulative_aid.flatten()
        squared_sum = np.sum(cell_aid ** 2)
        n_cells = np.sum(cell_aid > 0)
        
        if n_cells == 0:
            return 0.0
        
        jain = (total_aid ** 2) / (n_cells * squared_sum) if squared_sum > 0 else 0.0
        return float(jain)
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(self._grid, cmap='YlOrRd', vmin=0, vmax=self.max_demand)
        ax1.set_title(f'Remaining Demand (Step {self._step_count})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, label='Demand')
        
        ax2.imshow(self._grid, cmap='YlOrRd', vmin=0, vmax=self.max_demand, alpha=0.5)
        
        colors = ['red', 'blue', 'green']
        markers = ['o', 's', '^']
        for i, agent in enumerate(self.agents):
            x, y = self._agent_positions[agent]
            ax2.plot(y, x, marker=markers[i], color=colors[i], markersize=15, 
                    label=agent, markeredgecolor='black', markeredgewidth=2)
        
        ax2.set_title('Drone Positions')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('X')
        ax2.legend()
        
        plt.tight_layout()
        
        if self.render_mode == "rgb_array":
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = plt.imread(buf)
            plt.close()
            return (img * 255).astype(np.uint8)
        else:
            plt.show()
            plt.close()
            return None
    
    def close(self):
        pass
    
    def get_global_state(self) -> np.ndarray:
        state = self._grid.copy().astype(np.float32) / self.max_demand
        
        for i, agent in enumerate(self.agents):
            x, y = self._agent_positions[agent]
            state[x, y] = 1.0 + i * 0.1
        
        return state


if __name__ == "__main__":
    env = EquitableSwarmEnv(random_seed=42)
    print("Environment smoke test:")
    print(f"Agents: {env.agents}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, infos = env.reset(seed=42)
    print(f"\nReset observation shape for drone_0: {obs['drone_0'].shape}")
    
    total_reward = 0
    for step in range(10):
        actions = {agent: np.random.randint(5) for agent in env.agents}
        obs, rewards, dones, truncs, infos = env.step(actions)
        total_reward += rewards['drone_0']
        jain = infos['drone_0']['jain_index']
        print(f"Step {step+1}: reward={rewards['drone_0']:.4f}, jain={jain:.4f}")
        
        if all(dones.values()):
            break
    
    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
    print("Smoke test passed!")
