import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import csv
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter

from environment import EquitableSwarmEnv


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        flat_obs_dim = np.prod(obs_shape)
        
        layers = []
        prev_dim = flat_obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_layers = nn.Sequential(*layers)
        
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        
        obs_flat = obs.reshape(batch_size, -1)
        shared_features = self.shared_layers(obs_flat)
        
        logits = self.policy_head(shared_features)
        values = self.value_head(shared_features).squeeze(-1)
        
        return logits, values


class RolloutBuffer:
    def __init__(self, capacity: int, num_agents: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        
        self.observations = np.zeros((capacity, num_agents) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.log_probs = np.zeros((capacity, num_agents), dtype=np.float32)
        self.values = np.zeros((capacity, num_agents), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.float32)
        self.advantages = np.zeros((capacity, num_agents), dtype=np.float32)
        self.returns = np.zeros((capacity, num_agents), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, actions: np.ndarray, log_probs: np.ndarray, 
             values: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        idx = self.ptr % self.capacity
        
        self.observations[idx] = obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        last_value = self.values[(self.ptr - 1) % self.capacity] if self.ptr > 0 else 0
        last_advantage = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            self.advantages[step] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            last_advantage = self.advantages[step]
        
        self.returns = self.advantages + self.values
    
    def get_minibatch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            'observations': torch.tensor(self.observations[indices], dtype=torch.float32),
            'actions': torch.tensor(self.actions[indices], dtype=torch.long),
            'old_log_probs': torch.tensor(self.log_probs[indices], dtype=torch.float32),
            'advantages': torch.tensor(self.advantages[indices], dtype=torch.float32),
            'returns': torch.tensor(self.returns[indices], dtype=torch.float32)
        }


class PPOTrainer:
    def __init__(
        self,
        env: EquitableSwarmEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 4096,
        minibatch_size: int = 512,
        epochs_per_update: int = 10,
        random_seed: int = 42
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.epochs_per_update = epochs_per_update
        self.random_seed = random_seed
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.num_agents = len(env.agents)
        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCriticNetwork(self.obs_shape, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer(batch_size, self.num_agents, self.obs_shape)
        
        self.metrics_file = 'metrics.csv'
        self.writer = SummaryWriter(log_dir='runs/swarm_training')
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'mean_episode_reward', 'mean_jain_index', 'total_policy_loss', 'total_value_loss'])
    
    def _log_metrics(self, epoch: int, mean_reward: float, mean_jain: float, 
                     policy_loss: float, value_loss: float):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{mean_reward:.6f}", f"{mean_jain:.6f}", 
                           f"{policy_loss:.6f}", f"{value_loss:.6f}"])
        
        self.writer.add_scalar('Metrics/Episode_Reward', mean_reward, epoch)
        self.writer.add_scalar('Metrics/Jain_Fairness_Index', mean_jain, epoch)
        self.writer.add_scalar('Loss/Policy_Loss', policy_loss, epoch)
        self.writer.add_scalar('Loss/Value_Loss', value_loss, epoch)
        self.writer.add_scalar('Loss/Total_Loss', policy_loss + self.value_coef * value_loss, epoch)
        self.writer.flush()
    
    def collect_rollout(self) -> Tuple[float, float]:
        obs, _ = self.env.reset(seed=self.random_seed + self.global_step // self.batch_size)
        self.global_step += self.batch_size
        
        episode_rewards = []
        episode_jains = []
        current_episode_reward = 0.0
        current_episode_jains = []
        
        for step in range(self.batch_size):
            batch_obs = np.array([obs[agent] for agent in self.env.agents])
            obs_tensor = torch.from_numpy(batch_obs).float().to(self.device)
            
            with torch.no_grad():
                logits, values = self.network(obs_tensor)
            
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()
            
            obs_next, rewards, dones, truncs, infos = self.env.step(
                {agent: actions_np[i] for i, agent in enumerate(self.env.agents)}
            )
            
            rewards_np = np.array([rewards[agent] for agent in self.env.agents])
            dones_np = np.array([dones[agent] for agent in self.env.agents], dtype=float)
            
            self.buffer.add(
                batch_obs,
                actions_np,
                log_probs_np,
                values_np,
                rewards_np,
                dones_np
            )
            
            for i, agent in enumerate(self.env.agents):
                current_episode_reward += rewards_np[i]
                current_episode_jains.append(infos[agent]['jain_index'])
            
            if any(dones.values()):
                episode_rewards.append(current_episode_reward)
                episode_jains.append(np.mean(current_episode_jains))
                current_episode_reward = 0.0
                current_episode_jains = []
                obs, _ = self.env.reset(seed=self.random_seed + self.global_step // self.batch_size)
            else:
                obs = obs_next
        
        self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_jain = np.mean(episode_jains) if episode_jains else 0.0
        
        return mean_reward, mean_jain
    
    def update(self) -> Tuple[float, float]:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        indices = np.random.permutation(self.buffer.size)
        
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for start in range(0, self.buffer.size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                
                batch = self.buffer.get_minibatch(mb_indices)
                
                observations = batch['observations'].reshape(-1, *self.obs_shape).to(self.device)
                actions = batch['actions'].reshape(-1).to(self.device)
                old_log_probs = batch['old_log_probs'].reshape(-1).to(self.device)
                advantages = batch['advantages'].reshape(-1).to(self.device)
                returns = batch['returns'].reshape(-1).to(self.device)
                
                logits, values = self.network(observations)
                values = values.squeeze(-1)
                
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        num_updates = self.epochs_per_update * (self.buffer.size // self.minibatch_size)
        total_policy_loss /= num_updates
        total_value_loss /= num_updates
        
        return total_policy_loss, total_value_loss
    
    def train(self, num_epochs: int = 500):
        self.global_step = 0
        start_time = time.time()
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.network.parameters())}")
        print("-" * 80)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            mean_reward, mean_jain = self.collect_rollout()
            policy_loss, value_loss = self.update()
            
            self._log_metrics(epoch, mean_reward, mean_jain, policy_loss, value_loss)
            
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch:4d}/{num_epochs} | "
                  f"Reward: {mean_reward:8.4f} | "
                  f"Jain: {mean_jain:6.4f} | "
                  f"Policy Loss: {policy_loss:8.4f} | "
                  f"Value Loss: {value_loss:8.4f} | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"Elapsed: {elapsed:6.1f}s")
        
        torch.save(self.network.state_dict(), 'swarm_brain.pth')
        self.writer.close()
        print("-" * 80)
        print(f"Training completed in {time.time() - start_time:.1f} seconds")
        print(f"Model saved to swarm_brain.pth")
        print(f"TensorBoard logs saved to: runs/swarm_training")


def main():
    env = EquitableSwarmEnv(random_seed=42)
    trainer = PPOTrainer(
        env=env,
        lr=2e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.005,
        value_coef=0.5,
        batch_size=4096,
        minibatch_size=1024,
        epochs_per_update=2,
        random_seed=42
    )
    
    print("=" * 80)
    print("ULTRA-FAST TRAINING MODE - Optimized for 10-minute completion")
    print("Reduced epochs: 150 (vs 500), Batch size: 4096 (larger)")
    print("Increased learning rate: 2e-3 (vs 3e-4)")
    print("Reduced update epochs: 2 (vs 10) for faster learning")
    print("Reduced network size: [128, 64] (vs [256, 128, 64])")
    print("Reduced entropy coefficient: 0.005 for more focused learning")
    print("=" * 80)
    print()
    
    trainer.train(num_epochs=150)


if __name__ == "__main__":
    main()
