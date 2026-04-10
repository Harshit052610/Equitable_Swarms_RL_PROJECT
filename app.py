import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pathlib import Path


st.set_page_config(
    page_title="Equitable-Swarm: Disaster Relief",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


plt.style.use('grayscale')


def load_training_data():
    """Load training metrics from CSV file."""
    if os.path.exists('metrics.csv'):
        return pd.read_csv('metrics.csv')
    return None


def load_tensorboard_logs(log_dir='runs/swarm_training'):
    """Load TensorBoard event files for training evidence."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        logs = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            logs[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
        return logs
    except:
        return {}


def plot_training_metrics(df):
    """Create training metrics plots in black and white."""
    if df is None:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    epochs = df['epoch']
    
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['mean_episode_reward'], color='black', linewidth=2, label='Mean Reward')
    ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Aid Delivered', fontsize=12, fontweight='bold')
    ax1.set_title('Episode Reward Over Training', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['mean_jain_index'], color='black', linewidth=2, label="Jain's Index")
    ax2.axhline(y=0.8, color='black', linestyle='--', linewidth=1.5, label='Target (0.8)')
    ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Jain's Fairness Index", fontsize=12, fontweight='bold')
    ax2.set_title('Equity Distribution Over Training', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    ax3 = axes[1, 0]
    if 'total_policy_loss' in df.columns:
        ax3.plot(epochs, df['total_policy_loss'], color='black', linewidth=2, label='Policy Loss')
    ax3.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Policy Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Policy Loss Convergence', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()
    
    ax4 = axes[1, 1]
    if 'total_value_loss' in df.columns:
        ax4.plot(epochs, df['total_value_loss'], color='black', linewidth=2, label='Value Loss')
    ax4.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Value Loss', fontsize=12, fontweight='bold')
    ax4.set_title('Value Loss Convergence', fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend()
    
    plt.tight_layout()
    return fig


def plot_tensorboard_evidence(logs):
    """Create comprehensive training evidence from TensorBoard logs."""
    if not logs:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('Training Evidence: Complete Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metric_mapping = {
        'Metrics/Episode_Reward': (axes[0, 0], 'Episode Reward'),
        'Metrics/Jain_Fairness_Index': (axes[0, 1], "Jain's Fairness Index"),
        'Loss/Policy_Loss': (axes[0, 2], 'Policy Loss'),
        'Loss/Value_Loss': (axes[1, 0], 'Value Loss'),
        'Loss/Total_Loss': (axes[1, 1], 'Total Loss'),
    }
    
    for tag, (ax, title) in metric_mapping.items():
        if tag in logs:
            steps = logs[tag]['steps']
            values = logs[tag]['values']
            ax.plot(steps, values, color='black', linewidth=2)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.fill_between(steps, values, alpha=0.15, color='gray')
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Training Complete\n500 Epochs', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def display_training_summary(df, logs):
    """Display comprehensive training evidence summary."""
    st.markdown("### Training Evidence Summary")
    st.markdown("---")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Epochs", len(df))
        
        with col2:
            final_reward = df['mean_episode_reward'].iloc[-1]
            st.metric("Final Reward", f"{final_reward:.2f}")
        
        with col3:
            final_jain = df['mean_jain_index'].iloc[-1]
            st.metric("Final Fairness", f"{final_jain:.3f}")
        
        with col4:
            reward_growth = ((df['mean_episode_reward'].iloc[-1] - 
                             df['mean_episode_reward'].iloc[0]) / 
                             df['mean_episode_reward'].iloc[0] * 100)
            st.metric("Reward Growth", f"{reward_growth:.1f}%")
    
    st.markdown("---")
    
    if logs:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Metrics/Episode_Reward' in logs:
                rewards = logs['Metrics/Episode_Reward']['values']
                st.metric("Best Reward", f"{max(rewards):.2f}")
                st.metric("Avg Reward (Last 10)", f"{np.mean(rewards[-10:]):.2f}")
        
        with col2:
            if 'Metrics/Jain_Fairness_Index' in logs:
                jains = logs['Metrics/Jain_Fairness_Index']['values']
                st.metric("Best Fairness", f"{max(jains):.3f}")
                st.metric("Avg Fairness (Last 10)", f"{np.mean(jains[-10:]):.3f}")
        
        with col3:
            if 'Loss/Total_Loss' in logs:
                losses = logs['Loss/Total_Loss']['values']
                st.metric("Final Loss", f"{losses[-1]:.3f}")
                loss_reduction = ((losses[0] - losses[-1]) / losses[0] * 100)
                st.metric("Loss Reduction", f"{loss_reduction:.1f}%")


def main():
    st.title("Equitable-Swarm: Disaster Relief Operations")
    st.markdown("Multi-Agent Reinforcement Learning for Equitable Disaster Response")
    st.markdown("---")
    
    training_data = load_training_data()
    tensorboard_logs = load_tensorboard_logs()
    
    st.markdown("### Disaster Relief Operation Simulation")
    
    if os.path.exists('swarm_simulation.gif'):
        import base64
        with open('swarm_simulation.gif', 'rb') as f:
            gif_data = base64.b64encode(f.read()).decode()
        
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f'''
        <img src="data:image/gif;base64,{gif_data}" style="width:100%; max-width:1200px; height:auto; border:1px solid #ddd;">
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Simulation GIF not found. Please run analytics.py to generate visualizations.")
    
    st.markdown("---")
    st.markdown("### Training Evidence")
    st.markdown("Comprehensive analysis of model training performance")
    st.markdown("---")
    
    display_training_summary(training_data, tensorboard_logs)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Training Metrics", "Detailed Analysis"])
    
    with tab1:
        st.markdown("### Training Progress Visualization")
        
        if training_data is not None:
            fig = plot_training_metrics(training_data)
            if fig:
                st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Training data not available.")
        
        st.markdown("---")
        
        if tensorboard_logs:
            fig_tb = plot_tensorboard_evidence(tensorboard_logs)
            if fig_tb:
                st.pyplot(fig_tb, use_container_width=True)
        else:
            st.warning("TensorBoard logs not available.")
    
    with tab2:
        st.markdown("### Detailed Performance Analysis")
        
        if training_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Reward Distribution")
                fig_reward, ax = plt.subplots(figsize=(8, 5))
                ax.hist(training_data['mean_episode_reward'], bins=20, 
                       color='black', alpha=0.7, edgecolor='white')
                ax.set_xlabel('Reward', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title('Distribution of Rewards Across Epochs', fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_reward)
                plt.close()
            
            with col2:
                st.markdown("#### Fairness Distribution")
                fig_jain, ax = plt.subplots(figsize=(8, 5))
                ax.hist(training_data['mean_jain_index'], bins=20, 
                       color='black', alpha=0.7, edgecolor='white')
                ax.set_xlabel("Jain's Index", fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title("Distribution of Fairness Across Epochs", fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_jain)
                plt.close()
            
            st.markdown("---")
            st.markdown("#### Training Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Total Epochs', 'Initial Reward', 'Final Reward', 
                          'Initial Fairness', 'Final Fairness',
                          'Best Reward', 'Best Fairness'],
                'Value': [
                    len(training_data),
                    f"{training_data['mean_episode_reward'].iloc[0]:.2f}",
                    f"{training_data['mean_episode_reward'].iloc[-1]:.2f}",
                    f"{training_data['mean_jain_index'].iloc[0]:.3f}",
                    f"{training_data['mean_jain_index'].iloc[-1]:.3f}",
                    f"{training_data['mean_episode_reward'].max():.2f}",
                    f"{training_data['mean_jain_index'].max():.3f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Training data not available for detailed analysis.")
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("**Architecture**: Multi-Agent PPO with Actor-Critic Networks")
    st.markdown("**Environment**: 10x10 Grid World with 3 Autonomous Drones")
    st.markdown("**Objective**: Equitable aid distribution using Jain's Fairness Index")
    st.markdown("**Training Duration**: 861 seconds (500 epochs)")


if __name__ == "__main__":
    main()
