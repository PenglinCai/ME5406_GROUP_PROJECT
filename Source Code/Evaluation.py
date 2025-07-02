import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the output directory exists
os.makedirs('plots', exist_ok=True)

# File path configuration
FILES = {
    'DDPG_OU':  'Training Data/step_log_DDPG_OU_20250424-205228.csv',
    'SAC':      'Training Data/step_log_SAC_20250424-204153.csv',
    'DDPG_GAU': 'Training Data/step_log_DDPG_GAU_20250425-015709.csv',
    'SA4C':     'Training Data/step_log_SA4C_20250425-015306.csv'
}

# Moving-average window size
WINDOW = 40


def plot_total_reward(files, window=WINDOW):
    """
    Moving average of total reward per episode
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        ep_reward = df.groupby('episode')['reward'].sum()
        ma = ep_reward.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Reward vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/total_reward.png')
    plt.close(fig)


def plot_episode_length(files, window=WINDOW):
    """
    Moving average of episode length (in steps)
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        ep_len = df.groupby('episode')['step'].max()
        ma = ep_len.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'Episode Length vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/episode_length.png')
    plt.close(fig)


def plot_success_rate(files, window=WINDOW, threshold=50):
    """
    Success rate (moving average): episodes where on_goal reaches the given threshold
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        success = df.groupby('episode')['on_goal'].max().ge(threshold).astype(int)
        ma = success.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate vs Episode (on_goal ≥ {threshold}, Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/success_rate.png')
    plt.close(fig)


def plot_actor_loss(files, window=WINDOW):
    """
    Moving average of actor-network loss
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        aloss = df.groupby('episode')['actor_loss'].mean()
        ma = aloss.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.title(f'Actor Loss vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/actor_loss.png')
    plt.close(fig)


def plot_critic_loss(files, window=WINDOW):
    """
    Moving average of critic-network loss  
    – For SAC: use (critic1_loss + critic2_loss) / 2  
    – For SA4C: use (critic1_loss … critic4_loss) / 4  
    – For other algorithms: use 'critic_loss'
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        if 'SA4C' in name:
            cl_df = df.groupby('episode')[
                ['critic1_loss', 'critic2_loss', 'critic3_loss', 'critic4_loss']
            ].mean()
            avg_cl = cl_df.mean(axis=1)
        elif 'SAC' in name:
            cl1 = df.groupby('episode')['critic1_loss'].mean()
            cl2 = df.groupby('episode')['critic2_loss'].mean()
            avg_cl = (cl1 + cl2) / 2
        else:
            avg_cl = df.groupby('episode')['critic_loss'].mean()
        ma = avg_cl.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.title(f'Critic Loss vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/critic_loss.png')
    plt.close(fig)


def plot_q_value(files, window=WINDOW):
    """
    Moving average of the average Q-value
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        qv = df.groupby('episode')['q_value'].mean()
        ma = qv.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')
    plt.title(f'Average Q-Value vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/average_q_value.png')
    plt.close(fig)


def plot_collision_rate_all(files, window=WINDOW, threshold=0.005):
    """
    Collision rate (moving average): mini_distance < threshold
    """
    fig = plt.figure(figsize=(8, 4))
    for name, path in files.items():
        df = pd.read_csv(path)
        collision = df.groupby('episode')['min_distance'].min().lt(threshold).astype(int)
        ma = collision.rolling(window, min_periods=1).mean()
        plt.plot(ma.index, ma.values, label=name)
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate')
    plt.ylim(0, 1)
    plt.title(f'Collision Rate vs Episode (Moving Avg, window={window})')
    plt.legend()
    plt.grid(True)
    fig.savefig('plots/collision_rate.png')
    plt.close(fig)


if __name__ == '__main__':
    plot_total_reward(FILES)
    plot_episode_length(FILES)
    plot_success_rate(FILES, threshold=50)
    plot_actor_loss(FILES)
    plot_critic_loss(FILES)
    plot_q_value(FILES)
    plot_collision_rate_all(FILES, window=WINDOW, threshold=0.0001)
