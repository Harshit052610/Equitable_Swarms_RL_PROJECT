import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8

def save_ieee_plot(data, x_col, y_col, title, ylabel, filename, color='black', target=None):
    plt.figure(figsize=(5, 4))
    plt.plot(data[x_col], data[y_col], color=color, linewidth=1.5, label=ylabel)
    
    if target is not None:
        plt.axhline(y=target, color='black', linestyle='--', linewidth=1, label=f'Target ({target})')
    
    plt.title(title, fontsize=11, fontweight='bold')
    plt.xlabel('Training Epoch', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

df = pd.read_csv('metrics.csv')

print("=" * 80)
print("IEEE-FORMAT PUBLICATION QUALITY GRAPHS")
print("=" * 80)
print(f"\nData loaded: {len(df)} epochs")
print(f"Epoch range: {df['epoch'].min()} to {df['epoch'].max()}")

df_clean = df.dropna(subset=['mean_episode_reward', 'mean_jain_index', 'total_policy_loss', 'total_value_loss'])

print("\n--- Generating IEEE-Format Graphs ---\n")

save_ieee_plot(df_clean, 'epoch', 'mean_episode_reward', 
              'Episode Reward Over Training', 
              'Total Aid Delivered', 
              'fig_reward.png')

save_ieee_plot(df_clean, 'epoch', 'mean_jain_index', 
              'Equity Distribution Over Training', 
              "Jain's Fairness Index", 
              'fig_jain.png', 
              target=0.8)

save_ieee_plot(df_clean, 'epoch', 'total_policy_loss', 
              'Policy Loss Convergence', 
              'Policy Loss', 
              'fig_ploss.png')

save_ieee_plot(df_clean, 'epoch', 'total_value_loss', 
              'Value Loss Convergence', 
              'Value Loss', 
              'fig_vloss.png')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(df_clean['epoch'], df_clean['mean_episode_reward'], 'black', linewidth=1.5)
axes[0, 0].set_title('Episode Reward', fontsize=10, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=9)
axes[0, 0].set_ylabel('Reward', fontsize=9)
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

axes[0, 1].plot(df_clean['epoch'], df_clean['mean_jain_index'], 'black', linewidth=1.5)
axes[0, 1].axhline(y=0.8, color='black', linestyle='--', linewidth=1, label='Target (0.8)')
axes[0, 1].set_title("Jain's Fairness Index", fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=9)
axes[0, 1].set_ylabel('Fairness Index', fontsize=9)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].legend(fontsize=8)

axes[1, 0].plot(df_clean['epoch'], df_clean['total_policy_loss'], 'black', linewidth=1.5)
axes[1, 0].set_title('Policy Loss', fontsize=10, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=9)
axes[1, 0].set_ylabel('Loss', fontsize=9)
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

axes[1, 1].plot(df_clean['epoch'], df_clean['total_value_loss'], 'black', linewidth=1.5)
axes[1, 1].set_title('Value Loss', fontsize=10, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=9)
axes[1, 1].set_ylabel('Loss', fontsize=9)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Multi-Agent PPO Training Metrics: Equitable Disaster Relief', 
             fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('fig_all_metrics.png', dpi=300, bbox_inches='tight')
print(f"Saved: fig_all_metrics.png (combined view)")
plt.close()

print("\n" + "=" * 80)
print("IEEE-PUBLICATION QUALITY GRAPHS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated files:")
print("  1. fig_reward.png     - Episode reward over 500 epochs")
print("  2. fig_jain.png       - Jain's Fairness Index with target")
print("  3. fig_ploss.png      - Policy loss convergence")
print("  4. fig_vloss.png      - Value loss convergence")
print("  5. fig_all_metrics.png - Combined 4-panel overview")
print("\nAll images are 300 DPI, suitable for IEEE publication.")
print("=" * 80)
