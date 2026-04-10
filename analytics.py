import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import csv
from typing import List, Tuple
import torch

from environment import EquitableSwarmEnv
from train_ppo import ActorCriticNetwork


def load_metrics(csv_path: str) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    epochs, rewards, jains, policy_losses, value_losses = [], [], [], [], []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            rewards.append(float(row['mean_episode_reward']))
            jains.append(float(row['mean_jain_index']))
            policy_losses.append(float(row['total_policy_loss']))
            value_losses.append(float(row['total_value_loss']))
    
    return epochs, rewards, jains, policy_losses, value_losses


def smooth_curve(data: List[float], window: int = 10) -> np.ndarray:
    data_array = np.array(data)
    if len(data_array) < window:
        return data_array
    
    kernel = np.ones(window) / window
    smoothed = np.convolve(data_array, kernel, mode='same')
    
    edge = window // 2
    smoothed[:edge] = data_array[:edge]
    smoothed[-edge:] = data_array[-edge:]
    
    return smoothed


def plot_reward_curve(epochs: List[int], rewards: List[float], output_path: str = 'reward_curve.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    smoothed_rewards = smooth_curve(rewards, window=10)
    
    ax.plot(epochs, rewards, alpha=0.3, color='#2E86AB', linewidth=1, label='Raw Data')
    ax.plot(epochs, smoothed_rewards, color='#A23B72', linewidth=3, label='Trend (Smoothed)')
    
    ax.fill_between(epochs, rewards, smoothed_rewards, alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Aid Delivered', fontsize=13, fontweight='bold')
    ax.set_title('Disaster Relief Effectiveness Over Training', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved reward curve to {output_path}")


def plot_fairness_curve(epochs: List[int], jains: List[float], output_path: str = 'fairness_curve.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    smoothed_jains = smooth_curve(jains, window=10)
    
    ax.plot(epochs, jains, alpha=0.3, color='#F18F01', linewidth=1, label='Raw Data')
    ax.plot(epochs, smoothed_jains, color='#C73E1D', linewidth=3, label='Trend (Smoothed)')
    
    ax.fill_between(epochs, jains, smoothed_jains, alpha=0.2, color='#F18F01')
    
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='Target Equity (0.8)')
    
    ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel("Jain's Fairness Index", fontsize=13, fontweight='bold')
    ax.set_title('Equity Distribution Across Disaster Zones', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1])
    
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved fairness curve to {output_path}")


def create_advanced_swarm_animation(
    model_path: str = 'swarm_brain.pth',
    output_path: str = 'swarm_simulation.gif',
    max_steps: int = 150,
    fps: int = 8,
    seed: int = 42
):
    print("Creating advanced disaster relief simulation...")
    
    env = EquitableSwarmEnv(random_seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    
    network = ActorCriticNetwork(obs_shape, action_dim).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    
    obs, _ = env.reset(seed=seed)
    
    history = {
        'demand_grid': [env._grid.copy()],
        'agent_positions': [{agent: pos for agent, pos in env._agent_positions.items()}],
        'rewards': [{agent: 0 for agent in env.agents}],
        'aid_delivered': [0]
    }
    
    total_aid = 0
    
    for step in range(max_steps):
        batch_obs = np.array([obs[agent] for agent in env.agents])
        obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits, _ = network(obs_tensor)
        
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        actions_np = actions.cpu().numpy()
        
        obs, rewards, dones, truncs, infos = env.step(
            {agent: actions_np[i] for i, agent in enumerate(env.agents)}
        )
        
        step_aid = sum(rewards.values())
        total_aid += step_aid
        
        history['demand_grid'].append(env._grid.copy())
        history['agent_positions'].append({agent: pos for agent, pos in env._agent_positions.items()})
        history['rewards'].append(rewards.copy())
        history['aid_delivered'].append(total_aid)
        
        if any(dones.values()):
            break
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[:, :2])
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_aid = fig.add_subplot(gs[1, 2])
    
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    drone_names = ['Alpha', 'Beta', 'Gamma']
    
    demand_cmap = LinearSegmentedColormap.from_list('demand', 
        ['#FFB703', '#FB8500', '#D00000', '#7A0000'], N=10)
    
    aid_history = []
    
    def animate(frame):
        ax_main.clear()
        ax_stats.clear()
        ax_aid.clear()
        
        demand_grid = history['demand_grid'][frame]
        positions = history['agent_positions'][frame]
        rewards_step = history['rewards'][frame]
        
        ax_main.imshow(demand_grid, cmap=demand_cmap, vmin=0, vmax=9, origin='upper')
        
        for i in range(10):
            for j in range(10):
                if demand_grid[i, j] > 0:
                    alpha = 0.3 + (demand_grid[i, j] / 9) * 0.7
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                    linewidth=2, edgecolor='black', 
                                    facecolor=demand_cmap(demand_grid[i, j] / 9),
                                    alpha=alpha)
                    ax_main.add_patch(rect)
                    
                    text_color = 'white' if demand_grid[i, j] > 4 else 'black'
                    ax_main.text(j, i, str(demand_grid[i, j]),
                                ha="center", va="center", color=text_color, 
                                fontsize=11, fontweight='bold')
        
        for i, agent in enumerate(env.agents):
            x, y = positions[agent]
            
            outer_circle = Circle((y, x), 0.6, color=colors[i], alpha=0.3, zorder=1)
            ax_main.add_patch(outer_circle)
            
            inner_circle = Circle((y, x), 0.35, color=colors[i], zorder=2)
            ax_main.add_patch(inner_circle)
            
            ax_main.text(y, x, drone_names[i][0], ha="center", va="center", 
                        color='white', fontsize=14, fontweight='bold', zorder=3)
            
            if frame > 0:
                traj_x = [history['agent_positions'][t][agent][0] for t in range(frame + 1)]
                traj_y = [history['agent_positions'][t][agent][1] for t in range(frame + 1)]
                ax_main.plot(traj_y, traj_x, color=colors[i], linewidth=3, 
                            alpha=0.7, zorder=1)
                
                for t in range(0, frame, 5):
                    ax_main.plot(traj_y[t], traj_x[t], marker='o', 
                                markersize=6, color=colors[i], alpha=0.4, zorder=1)
        
        current_aid = history['aid_delivered'][frame]
        aid_history.append(current_aid)
        
        ax_main.set_title(f'Disaster Relief Operations - Step {frame}/{len(history["demand_grid"])-1}', 
                         fontsize=16, fontweight='bold', pad=15)
        ax_main.set_xlabel('Grid Y (East-West)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Grid X (North-South)', fontsize=12, fontweight='bold')
        ax_main.set_xlim(-1, 10)
        ax_main.set_ylim(-1, 10)
        ax_main.grid(True, alpha=0.2, linestyle='--')
        
        drone_rewards = [rewards_step[agent] for agent in env.agents]
        colors_pie = ['#E63946', '#457B9D', '#2A9D8F']
        
        if sum(drone_rewards) > 0:
            explode = [0.1 if r > 0 else 0 for r in drone_rewards]
            wedges, texts, autotexts = ax_stats.pie(drone_rewards, explode=explode, 
                                                    labels=drone_names, colors=colors_pie,
                                                    autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
                                                    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax_stats.text(0.5, 0.5, 'No aid delivered yet', 
                         ha='center', va='center', fontsize=12, style='italic')
        
        ax_stats.set_title(f'Aid Delivery by Drone\nTotal: {sum(drone_rewards)} units', 
                          fontsize=13, fontweight='bold')
        
        if len(aid_history) > 1:
            ax_aid.plot(range(len(aid_history)), aid_history, 
                       color='#2E86AB', linewidth=3, marker='o', markersize=4)
            ax_aid.fill_between(range(len(aid_history)), aid_history, 
                              alpha=0.3, color='#2E86AB')
        
        ax_aid.set_title('Cumulative Aid Delivered', fontsize=13, fontweight='bold')
        ax_aid.set_xlabel('Time Step', fontsize=11)
        ax_aid.set_ylabel('Total Units', fontsize=11)
        ax_aid.grid(True, alpha=0.3, linestyle='--')
        ax_aid.set_xlim(0, max(50, len(aid_history)))
        
        fig.patch.set_facecolor('#f0f4f8')
        ax_main.set_facecolor('#ffffff')
        ax_stats.set_facecolor('#ffffff')
        ax_aid.set_facecolor('#ffffff')
    
    num_frames = len(history['demand_grid'])
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000//fps, blit=False)
    
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer, dpi=100)
    plt.close()
    print(f"Saved advanced swarm simulation to {output_path}")
    return output_path


def create_coverage_heatmap_animation(
    model_path: str = 'swarm_brain.pth',
    output_path: str = 'coverage_map.gif',
    max_steps: int = 100,
    fps: int = 6,
    seed: int = 42
):
    print("Creating coverage heatmap animation...")
    
    env = EquitableSwarmEnv(random_seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    
    network = ActorCriticNetwork(obs_shape, action_dim).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    
    obs, _ = env.reset(seed=seed)
    
    visit_counts = np.zeros((10, 10))
    
    history = {'visit_counts': [visit_counts.copy()], 'demand': [env._grid.copy()]}
    
    for step in range(max_steps):
        batch_obs = np.array([obs[agent] for agent in env.agents])
        obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits, _ = network(obs_tensor)
        
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        actions_np = actions.cpu().numpy()
        
        obs, rewards, dones, truncs, infos = env.step(
            {agent: actions_np[i] for i, agent in enumerate(env.agents)}
        )
        
        for agent in env.agents:
            x, y = env._agent_positions[agent]
            visit_counts[x, y] += 1
        
        history['visit_counts'].append(visit_counts.copy())
        history['demand'].append(env._grid.copy())
        
        if any(dones.values()):
            break
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        visits = history['visit_counts'][frame]
        demand = history['demand'][frame]
        
        im1 = ax1.imshow(visits, cmap='YlGnBu', origin='upper')
        ax1.set_title(f'Drone Visit Frequency\nStep {frame}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Y', fontsize=12)
        ax1.set_ylabel('X', fontsize=12)
        
        for i in range(10):
            for j in range(10):
                if visits[i, j] > 0:
                    ax1.text(j, i, str(int(visits[i, j])), 
                            ha="center", va="center", color="white", 
                            fontsize=10, fontweight='bold')
        
        coverage = (visits > 0).astype(int)
        im2 = ax2.imshow(coverage, cmap='RdYlGn', origin='upper')
        ax2.set_title('Coverage Map\n(Green = Visited)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Y', fontsize=12)
        ax2.set_ylabel('X', fontsize=12)
        
        for i in range(10):
            for j in range(10):
                if coverage[i, j] == 1:
                    ax2.text(j, i, 'v', ha="center", va="center", 
                            color="white", fontsize=14, fontweight='bold')
        
        coverage_pct = np.sum(coverage) / 100 * 100
        fig.suptitle(f'Zone Coverage Analysis - {coverage_pct:.0f}% Covered', 
                    fontsize=16, fontweight='bold')
    
    num_frames = len(history['visit_counts'])
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=1000//fps, blit=False)
    
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer, dpi=100)
    plt.close()
    print(f"Saved coverage heatmap to {output_path}")
    return output_path


def main():
    print("Generating advanced disaster relief visualizations...\n")
    
    epochs, rewards, jains, policy_losses, value_losses = load_metrics('metrics.csv')
    
    plot_reward_curve(epochs, rewards)
    plot_fairness_curve(epochs, jains)
    
    print("\nCreating main swarm simulation...")
    create_advanced_swarm_animation(
        model_path='swarm_brain.pth',
        output_path='swarm_simulation.gif',
        max_steps=150,
        fps=8,
        seed=42
    )
    
    print("\nCreating coverage heatmap...")
    create_coverage_heatmap_animation(
        model_path='swarm_brain.pth',
        output_path='coverage_map.gif',
        max_steps=100,
        fps=6,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
