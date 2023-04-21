import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns 
        a tuple (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.eps = self.eps_start # Extra param for annealing

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        # Linear annealing
        if self.eps > self.eps_end:
          self.eps -= (self.eps_start - self.eps_end) / self.anneal_length

        # 0-1 random uniform sample
        U = random.random() 

        # Run forward pass and return optimal actions
        if exploit or U > self.eps:
          action_values = self.forward(observation)
          return torch.argmax(action_values, dim=1)
        else: 
          # Return random actions
          return torch.randint(
            low=0, 
            high=self.n_actions, 
            size=(observation.size(0), 1)
          )



def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    
    sample = memory.sample(batch_size=dqn.batch_size) # Sample from memory
    observations = torch.cat(sample[0]).to(device) # Observations tensor
    actions = torch.cat(sample[1]).to(device) # Actions tensor
    rewards = torch.cat(sample[3]).to(device) # Rewards tensor
    
    # Special handling for next_observations with terminal states
    next_non_terminal = [] # Array for non-terminal next_observations
    terminal_states = [] # Bool array terminal and non-terminal (used later)
    for i in sample[2]:
      if torch.is_tensor(i):
        next_non_terminal.append(i)
        terminal_states.append(False)
      else:
        terminal_states.append(True)

    # Next observations tensor
    next_observations = torch.cat(next_non_terminal).to(device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    action_values = dqn.forward(observations)
    q_values = torch.gather(action_values, 1, actions.unsqueeze(dim=1))
    

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    target_action_values = target_dqn.forward(next_observations)

    # Extract max values and convert to list
    max_target_values = torch.max(target_action_values, dim=1)[0].tolist()

    # Insert zero values for terminal state positions
    for i in range(len(terminal_states)):
      if terminal_states[i]:
        max_target_values.insert(i, 0)
    
    # Create targets tensor
    targets = torch.tensor(max_target_values).to(device)

    # Calculate Q value targets
    q_value_targets = rewards + target_dqn.gamma * targets

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
