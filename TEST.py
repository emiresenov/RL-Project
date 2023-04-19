import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
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

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        
        '''
        # ---------------------------------------
        # Testing things for figuring out the act function in DQN
        # ---------------------------------------
        print(obs)
        
        # Test forward pass to see what happens
        x = dqn.forward(obs)

        # Check forward pass output
        print(x)

        # produce argmax tensor
        #print(torch.argmax(x, dim=1))

        # Test implemented act function
        #arr = []
        #for i in range(25):
          #arr.append(dqn.act(obs).item())
        
        #print(arr)
        # Act function implemented and seems to be working properly
        '''
        
        # -------------------------------------
        # Below I test various things to figure out what to do in train.py
        # -------------------------------------
        while not terminated:
            steps += 1

            # TODO: Get action from DQN.
            action = dqn.act(obs).item()

            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            action_tensor = torch.tensor(action, device=device).reshape(1)
            reward_tensor = torch.tensor(reward, device=device).float().reshape(1) 
            memory.push(obs, action_tensor, next_obs, reward_tensor)
            
            # (Added) Update obs
            obs = next_obs

    # -------------------------------------
    # Below I test various things to figure out how to complete the optimize
    # function in dqn.py
    # -------------------------------------

    # What does memory.sample return?
    sample = memory.sample(50)
    print(sample[0]) # Observation tensors
    print(sample[1]) # Action tensors
    print(sample[2]) # Next observation tensors
    print(sample[3]) # Reward tensors

    # Concatenate tensors
    observations = torch.cat(sample[0]).to(device)
    actions = torch.cat(sample[1]).to(device)
    rewards = torch.cat(sample[3]).to(device)

    # How do we validate terminal state?
    # Answer: terminal next_obs is never preprocessed
    # Type for preprocessed next_obs: <class 'torch.Tensor'>
    # Type for unprocessed next_obs: <class 'numpy.ndarray'>
    # Need special handling for terminal types in next_observations
    # Can use torch.is_tensor(arg) method

    # Check types of next_obs (notice difference for terminal and non-terminal)
    #for i in sam[2]:
      #print(torch.is_tensor(i))
    
    # Idea: create terminal states array
    next_non_terminal = []
    terminal_states = []
    for i in sample[2]:
      if torch.is_tensor(i):
        next_non_terminal.append(i)
        terminal_states.append(False)
      else:
        terminal_states.append(True)

    # Next observations tensor
    next_observations = torch.cat(next_non_terminal).to(device)

    
    # Try running forward passes
    action_values = dqn.forward(observations)
    q_values = torch.gather(action_values, 1, actions.unsqueeze(dim=1))

    #print("Action values for observations:", action_values)
    #print("q-values for observations:", q_values)

    # Try running forward pass with next_observations (here I just try with dqn
    # instead of targetDQN since I just want to validate output formats)
    target_action_values = dqn.forward(next_observations)

    # Extract max values and convert to list
    max_target_values = torch.max(target_action_values, dim=1)[0].tolist()

    #print(target_action_values)
    #print(max_target_values)

    # Insert zero values for terminal states, see idea and tests below
    for i in range(len(terminal_states)):
      if terminal_states[i]:
        max_target_values.insert(i, 0)
    
    print(max_target_values)

    # Create targets tensor
    targets = torch.tensor(max_target_values).to(device)
    print("targets:", targets)

    print("rewards:", rewards)

    print("q-values:", q_values.squeeze())

    # Try to add reward tensor and target tensor (see if dimensions match)
    q_val_tar_test = rewards + 0.9*targets

    print("q val targets:", q_val_tar_test)

    print(q_values.squeeze().size()) # Check that dimensions match
    print(q_val_tar_test.size())    

    



    '''
    # Just tried different stuff here

    #print(torch.cat(next_observations).to(device))

    # Idea: After forward pass, insert zero tensors for terminal states, 
    # here is just a test of insert without forward pass
    for i in range(len(terminal_states)):
      if terminal_states[i]:
        next_observations.insert(i, torch.tensor(0, device=device).float().reshape(1))
    
    print(next_observations)
    assert(len(next_observations) == len(sam[3])) # Confirm length is correct
    # Confirm that insert works the way it should
    for i in range(len(next_observations)):
      if torch.is_tensor(sam[2][i]) == False:
          assert next_observations[i].item() == 0



    # Try concatenating tensor tuples
    observations = torch.cat(sam[0]).to(device)
    actions = torch.cat(sam[1]).to(device)
    #next_observations = torch.cat(sam[2]).to(device)
    #print(observations)
    #print(actions.unsqueeze(dim=1)) # Unsqueeze for torch gather
    #print(next_observations)
    '''


        
    # Close environment after training is completed.
    env.close()




