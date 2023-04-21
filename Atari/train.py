import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import copy # (Added) For Q' = Q step


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.Atari
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(
      env, 
      screen_size=84, 
      grayscale_obs=True, 
      frame_skip=1, 
      noop_max=30
    )
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    # (Added) Keep track of step count.
    steps = 0

    # (Added) Keep track of episode counts and mean returns
    episode_counts = []
    mean_returns = []

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        # Initialize frame stack tensor
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)
        
        while not terminated:

            # Only act every 4 steps
            if steps % env_config['obs_stack_size'] == 0:
              # TODO: Get action from DQN.
              action = dqn.act(obs_stack).item()

            # Act in the true environment.
            obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            if not terminated:
                obs = preprocess(obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs_stack = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            action_tensor = torch.tensor(action, device=device).reshape(1)
            reward_tensor = torch.tensor(reward, device=device).float().reshape(1) 
            memory.push(obs_stack, action_tensor, next_obs_stack, reward_tensor)

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps % env_config["train_frequency"] == 0:
              optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if steps & env_config["target_update_frequency"] == 0:
              target_dqn = copy.deepcopy(dqn)
            
            # (Added) Update obs
            obs_stack = next_obs_stack

            steps += 1
            
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            episode_counts.append(episode+1)
            mean_returns.append(mean_return)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'Atari/models/{args.env}_best.pt')

    # Output arrays for plotting
    print("Episodes array: ", episode_counts)
    print("Mean returns array: ", mean_returns)


    # Close environment after training is completed.
    env.close()
