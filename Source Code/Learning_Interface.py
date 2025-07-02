### Learning_Interface.py

from Environment import ArmEnv
# from Neural_Network_DDPG_GAU import DDPG_GAU_Agent
from Neural_Network_DDPG_OU import DDPG_OU_Agent
from Neural_Network_DDPG_GAU import DDPG_GAU_Agent
from Neural_Network_SAC import SACAgent
from Neural_Network_SA4C import SA4CAgent

import pandas as pd
import os
import time

MAX_EPISODES = 1000
MAX_EPISODES_STEPS = 300
BATCH_SIZE = 64
ON_TRAIN = True  # True: train; False: evaluate

# Select algorithm
ALGORITHM = "DDPG_OU"  # "DDPG_OU", "DDPG_GAU", "SAC", "SA4C"

# Initialize environment
env = ArmEnv()
state_dim = env.state_dim
action_dim = env.action_dim
action_bound = env.action_bound  # scalar or [low, high]

# Instantiate agent according to the chosen algorithm
if ALGORITHM.upper() == "SAC":
    rl = SACAgent(state_dim, action_dim, action_bound)
elif ALGORITHM.upper() == "DDPG_OU":
    rl = DDPG_OU_Agent(state_dim, action_dim, action_bound)
elif ALGORITHM.upper() == "DDPG_GAU":
    rl = DDPG_GAU_Agent(state_dim, action_dim, action_bound)
elif ALGORITHM.upper() == "SA4C":
    rl = SA4CAgent(state_dim, action_dim, action_bound)
else:
    raise ValueError(f"Unknown ALGORITHM {ALGORITHM}")


def train():
    log_dir = 'Training Data'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    step_log_path = os.path.join(log_dir, f'step_log_{ALGORITHM}_{timestamp}.csv')

    # Create an empty log file and write column names (executed only once)
    columns = ['episode', 'step', 'reward', 'pose_error', 'orient_error', 'min_distance',
               'on_goal', 'actor_loss', 'critic_loss',
               'critic1_loss', 'critic2_loss', 'critic3_loss', 'critic4_loss', 'q_value'] + \
              [f'joint_{k}' for k in range(7)] + [f'action_{k}' for k in range(7)]
    pd.DataFrame(columns=columns).to_csv(step_log_path, index=False)

    for ep in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0.0
        collision = False
        start = time.time()

        for step in range(MAX_EPISODES_STEPS):
            action = rl.select_action(state, evaluate=False)
            next_state, reward, done, pose_err, orient_err, min_dist = env.step(action)

            if min_dist <= 0.005:
                collision = True

            # Unified interface: TD3 ignores the extra 'done' argument
            rl.store_transition(state, action, reward, next_state, done)

            if ALGORITHM.upper() == "SAC":
                actor_loss, critic1_loss, critic2_loss = rl.update(BATCH_SIZE)
                critic_loss = critic3_loss = critic4_loss = 0
            elif ALGORITHM.upper() == "SA4C":
                actor_loss, critic1_loss, critic2_loss, critic3_loss, critic4_loss = rl.update(BATCH_SIZE)
                critic_loss = 0
            else:
                actor_loss, critic_loss = rl.update(BATCH_SIZE)
                critic1_loss = critic2_loss = critic3_loss = critic4_loss = 0

            q_value = rl.evaluate_q(state, action)

            ep_reward += reward
            state = next_state

            # Record each step's data
            step_data = {
                'episode': ep + 1,
                'step': step + 1,
                'reward': reward,
                'pose_error': pose_err,
                'orient_error': orient_err,
                'min_distance': min_dist,
                'on_goal': int(env.on_goal),
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'critic1_loss': critic1_loss,
                'critic2_loss': critic2_loss,
                'critic3_loss': critic3_loss,
                'critic4_loss': critic4_loss,
                'q_value': q_value,
            }
            for k in range(7):
                step_data[f'joint_{k}'] = env.arm_info[k]
                step_data[f'action_{k}'] = action[k]

            # Append the data to the CSV file
            pd.DataFrame([step_data]).to_csv(step_log_path, mode='a', header=False, index=False)

            if done or step == MAX_EPISODES_STEPS - 1:
                print(f"Ep:{ep + 1} | {'done' if done else '---'} | Coll:{int(collision)} | "
                      f"R:{ep_reward:.1f} | Step:{step + 1} | "
                      f"PoseErr:{pose_err:.4f} | OriErr:{orient_err:.4f}")
                break

        print("Time/ep:", time.time() - start)

    rl.save()


def eval():
    rl.load()
    print("Loaded trained parameters.")
    for ep in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0.0
        collision = False
        for step in range(MAX_EPISODES_STEPS):
            action = rl.select_action(state, evaluate=True)
            state, reward, done, pose_err, orient_err, min_dist = env.step(action)
            ep_reward += reward
            if done or step == MAX_EPISODES_STEPS - 1:
                print(f"Eval Ep:{ep + 1} | {'done' if done else '---'} | Coll:{int(collision)} | "
                      f"R:{ep_reward:.1f}")
                break


if ON_TRAIN:
    train()
else:
    eval()
