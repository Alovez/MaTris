from collections import namedtuple
import math
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from game.matris import Game
from itertools import count


env = Game()
TARGET_REPLACE_ITER = 100
BATCH_SIZE = 50
MEMORY_CAPACITY = 50000
EPSILON = 0.9
GAMMA = .999
N_STATES = 10
N_ACTIONS = env.n_actions
LR = .001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.line1 = nn.Linear(N_STATES, 50)
        self.line1.weight.data.normal_(0, .1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, .1)

    def forward(self, x):
        x = self.line1(x)
        x = F.relu(x)
        out = self.out(x)
        return out

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = NN().to(device), NN().to(device)
        self.learning_step_counter = 0
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # TODO: compare more optimizer
        self.loss_func = nn.MSELoss()  # TODO: compare more loss function

    def choose_action(self, x):
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self):
        if self.learning_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter += 1

        if len(self.memory) < BATCH_SIZE:
            return
        sample = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*sample))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.eval_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

dqn = DQN()


for i_episode in range(100000):
    s = env.reset()

    import ipdb; ipdb.set_trace()
    for t in count():
        a = dqn.choose_action(s)

        s_, r, done = env.step(a)

        dqn.memory.push(s, a, r, s_)

        dqn.learn()

        if done:
            break

        s = s_


