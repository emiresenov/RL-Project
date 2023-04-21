import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

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

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    steps = 0

    # ---------------------------------------
    # Testing things for figuring out preprocessing, stacking and act function
    # ---------------------------------------
    obs, info = env.reset()

    obs = preprocess(obs, env=args.env).unsqueeze(0)
    #print(obs.size()) # (1,84,84)

    # Testing Atari stacking and preprocessing
    obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)
    #print(obs_stack)
    #print(obs_stack.size()) #(1,4,84,84)

    # Test forward pass to see what happens
    x = dqn.forward(obs_stack)

    # Check forward pass output
    print(x) # it works

    # New problem: act has to output values 2 and 3 instead of 0 and 1
    #actions = torch.argmax(x, dim=1)
    #print(actions)
    # Idea: just add a two tensor to the argmax tensor
    #addTwoTensor = torch.Tensor([2]).to(device)
    #print(addTwoTensor + actions)

    # Added above adjustments to act, now try it
    #arr = []
    #for i in range(25):
      #arr.append(dqn.act(obs_stack).item())
    
    #print(arr) # seems to be working properly
    


    action = dqn.act(obs_stack).item() 

    # Try to take a step, append stack and act again
    for i in range(25):
      obs, reward, terminated, truncated, info = env.step(action)

      obs = preprocess(obs, env=args.env).unsqueeze(0)

      next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)

      obs_stack = next_obs_stack

      print(dqn.act(obs_stack).item())
        

    


        
    # Close environment after training is completed.
    env.close()




