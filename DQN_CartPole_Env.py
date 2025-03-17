import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device found: ", DEVICE)

# Hyperparameters
CONFIG = {
    "LEARNING_RATE": 0.001,
    "GAMMA": 0.99,
    "EPS_START": 1.0,
    "EPS_DECAY": 0.99,
    "EPS_END": 0.05,
    "NUM_ENVS": 5,
    "MAX_STEPS": 10**6,
    "INTERVAL": 10000,
    "TARGET_UPDATE": 1000,
    "BATCH_SIZE": 512,
    "REPLAY_BUFFER_SIZE": 1000000,
}


# Create environment
def make_env():
    return gym.make('CartPole-v1')

# Vectorize the Environment
def create_vector_env():
    return gym.vector.SyncVectorEnv([make_env for _ in range(CONFIG["NUM_ENVS"])])


# Define Q-Network
class CartPoleBrain(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(CartPoleBrain, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def act(action_space, local_qnetwork, states, epsilon):
    """ Implementation of actions """
    if np.random.rand() < epsilon:
        np.array([action_space for _ in range(CONFIG["NUM_ENVS"])])
    return local_qnetwork(torch.FloatTensor(states).to(DEVICE)).argmax(dim=1).cpu().numpy()

# Training function
def CartPoleAgent(env, use_target_qnetwork=False, use_experience_replay=False):
    state_size = env.single_observation_space.shape[0]
    num_actions = env.single_action_space.n
    local_qnetwork = CartPoleBrain(state_size, num_actions).to(DEVICE)
    target_qnetwork = None

    if use_target_qnetwork:
        target_qnetwork = CartPoleBrain(state_size, num_actions).to(DEVICE)
        target_qnetwork.load_state_dict(local_qnetwork.state_dict())

    optimizer = optim.Adam(local_qnetwork.parameters(), lr=CONFIG["LEARNING_RATE"])
    loss_func = nn.MSELoss()
    replay_buff = deque(maxlen=CONFIG["REPLAY_BUFFER_SIZE"]) if use_experience_replay or (use_target_qnetwork and use_experience_replay) else None

    epsilon = CONFIG["EPS_START"]
    ep_rewards = np.zeros(CONFIG["NUM_ENVS"])
    tot_rewards = []
    rewards_per_step = []
    states, _ = env.reset()

    for steps in range(CONFIG["MAX_STEPS"]):
        if steps % CONFIG["INTERVAL"] == 0:
            print(f"Step {steps}/{CONFIG['MAX_STEPS']}")
        action_space = env.single_action_space.sample()
        actions = act(action_space, local_qnetwork, states, epsilon)


        next_states, rewards, done, _, _ = env.step(actions)
        steps += CONFIG["NUM_ENVS"]

        if use_experience_replay or (use_target_qnetwork and use_experience_replay):
            for i in range(CONFIG["NUM_ENVS"]):
                replay_buff.append((states[i], actions[i], rewards[i], next_states[i], done[i]))

        if steps % 10 == 0:
            # Sampling from the replay buffer
            if use_experience_replay or (use_target_qnetwork and use_experience_replay) and len(replay_buff) >= CONFIG["BATCH_SIZE"]:
                experiences = random.choices(replay_buff, k=CONFIG["BATCH_SIZE"])
                states_exp, actions_exp, rewards_exp, next_states_exp, dones_exp = zip(*experiences)

                states_exp = torch.FloatTensor(states_exp).to(DEVICE)
                actions_exp = torch.LongTensor(actions_exp).to(DEVICE)
                rewards_exp = torch.FloatTensor(rewards_exp).to(DEVICE)
                next_states_exp = torch.FloatTensor(next_states_exp).to(DEVICE)
                dones_exp = torch.FloatTensor(dones_exp).to(DEVICE)
            else:
                states_exp = torch.FloatTensor(states).to(DEVICE)
                actions_exp = torch.LongTensor(actions).to(DEVICE)
                rewards_exp = torch.FloatTensor(rewards).to(DEVICE)
                next_states_exp = torch.FloatTensor(next_states).to(DEVICE)
                dones_exp = torch.FloatTensor(done).to(DEVICE)

            # Compute Q-values
            with torch.no_grad():
                next_q_values = (target_qnetwork(next_states_exp) if use_target_qnetwork else local_qnetwork(next_states_exp))
                if use_experience_replay or (use_target_qnetwork and use_experience_replay):
                    next_qMax = next_q_values.max(dim=1).values
                    targets = rewards_exp + CONFIG["GAMMA"] * next_qMax * (1 - dones_exp)
                else:
                    next_qMax = next_q_values.max(dim=1).values.cpu().numpy()
                    targets = torch.FloatTensor(rewards_exp + CONFIG["GAMMA"] * next_qMax * (~done)).to(DEVICE)

            if use_experience_replay or (use_target_qnetwork and use_experience_replay):
                predicted = local_qnetwork(states_exp)[range(CONFIG["BATCH_SIZE"]), actions_exp]
            else:
                predicted = local_qnetwork(states_exp)[range(CONFIG["NUM_ENVS"]), actions_exp]
            loss = loss_func(predicted, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_target_qnetwork or (use_target_qnetwork and use_experience_replay) and steps % CONFIG["TARGET_UPDATE"] == 0:
                target_qnetwork.load_state_dict(local_qnetwork.state_dict())

        states = next_states
        ep_rewards += rewards

        if np.any(done):
            for i in range(CONFIG["NUM_ENVS"]):
                if done[i]:
                    state, _ = env.envs[i].reset()
                    states[i] = state
                    tot_rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0

        epsilon = max(epsilon * CONFIG["EPS_DECAY"], CONFIG["EPS_END"])

        if steps % CONFIG["INTERVAL"] == 0:
            if use_target_qnetwork or use_experience_replay or (use_target_qnetwork and use_experience_replay) and tot_rewards:
                rewards_per_step.append(np.mean(tot_rewards[-100:]))
            else:
                rewards_per_step.append(np.mean(ep_rewards))

    return rewards_per_step



def implementation():
    env = create_vector_env()
    reps = 5
    variants = {
        "naive_q_learning": [],
        "only_tn": [],
        "only_er": [],
        "tn_n_er":[]
    }

    for r in range(reps):
        print(f"\nStarting Rep {r + 1}/{reps}\n")

        print("############## Training Naive Q learning ##############")
        variants["naive_q_learning"].append(CartPoleAgent(env, use_target_qnetwork= False, use_experience_replay= False))
        print("############### Training Only TN ##############")
        variants["only_tn"].append(CartPoleAgent(env, use_target_qnetwork= True, use_experience_replay= False))
        print("############## Training Only ER ##############")
        variants["only_er"].append(CartPoleAgent(env, use_target_qnetwork= False, use_experience_replay= True))
        print("############## Training TN and ER ##############")
        variants["tn_n_er"].append(CartPoleAgent(env, use_target_qnetwork= True, use_experience_replay= True))

    # Mean of the total rewards
    mean = {key: np.mean([r for r in reps if r], axis=0) for key, reps in variants.items()}
    # Standard deviation of the total rewards
    std_deviation = {key: np.std([r for r in reps if r], axis=0) for key, reps in variants.items()}
    steps = np.arange(0, len(mean["naive_q_learning"])) * (CONFIG["INTERVAL"] / 1000)

    # Plotting starts here
    plt.figure(figsize=(12, 8))
    window_size = 10
    for i in variants.keys():
        if mean[i].size > 0:
            # Smoothing to reduce noise
            smoothed_rewards = np.convolve(mean[i], np.ones(window_size) / window_size, mode='valid')
            padding_plt = np.linspace(0, smoothed_rewards[0], window_size - 1)
            smoothed_rewards = np.concatenate([padding_plt, smoothed_rewards])
            smoothed_steps = steps
            plt.plot(smoothed_steps, smoothed_rewards, label=i, marker='o', linestyle='-', linewidth=2)
            plt.fill_between(smoothed_steps, smoothed_rewards - std_deviation[i], smoothed_rewards + std_deviation[i], alpha=0.2)

    plt.xlabel('Steps (x1000)')
    plt.ylabel('Mean-Reward')
    plt.title('DQN Variants')
    plt.legend(title='Method')
    plt.grid(True)
    plt.ylim(0, 600)
    plt.xticks(np.arange(0, 1001, 100))
    plt.savefig("Output.png")
    plt.show()

    return variants

if __name__ == "__main__":
    implementation()
