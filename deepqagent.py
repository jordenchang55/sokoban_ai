import torch
import torch.nn as nn
import torch.nn.functional as funct
from agent import Agent
import numpy as np
from collections import deque
import random

from pathlib import Path
class SokobanNet(nn.Module):
    def __init__(self, xlim, ylim, dropout=0.5):
        super(SokobanNet, self).__init__()
        self.xlim = xlim
        self.ylim = ylim
        self.dropout = dropout

        channels = 4
        self.conv1 = nn.Conv2d(2, channels, kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)

        self.fc1 = nn.Linear(4*(xlim+1)*(ylim+1), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, 4)

    def forward(self, s):
        #s = s.view()

        s = funct.relu(self.bn1(self.conv1(s)))
        #print(s.size())
        s = funct.relu(self.bn2(self.conv2(s)))
        s = funct.relu(self.bn3(self.conv3(s)))
        s = funct.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, 4*(self.xlim+1)*(self.ylim+1))

        s = funct.dropout(funct.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = funct.dropout(funct.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        q = self.fc4(s)

        return q


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, sample_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=sample_size,
            replace=False
        )
        return [self.buffer[i] for i in index]

class DeepQAgent(Agent):



    def __init__(self, environment, discount_factor=0.95, minibatch_size = 32, verbose=False):
        super().__init__(environment)

        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        #if torch.cuda.is_available():
        self.model = SokobanNet(self.environment.xlim, self.environment.ylim)
        if torch.cuda.is_available():
            self.model.cuda()

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6)

        self.replay_buffer = ReplayBuffer(buffer_size = 500)


    def reward(self, state, action):
        box_position = np.array(self.environment.get_player(state)) + action
        next_box_position = box_position + action
        
        box_pushing = state[1, box_position[0], box_position[1]] == 1. and state[1, next_box_position[0], next_box_position[1]] == 0.
        push_on_goal = box_pushing and (tuple(next_box_position) in self.environment.storage)

        # not_scored = False
        # for i in range(len(self.environment.state[2:])):
        #   if (box_position == self.environment.state[2+i]).all():
        #       not_scored = not self.environment.has_scored[i]



        goal_reach = all([state[1, place[0], place[1]] == 1 for place in self.environment.storage])
        if push_on_goal:
            goal_reach = True
            set_difference = self.environment.storage.difference({tuple(next_box_position)})
            for place in set_difference:
                if state[1, place[0], place[1]] == 0:
                    goal_reach = False
        else:
            goal_reach = False

        state_hash = state.tobytes()
        if goal_reach:
            #print("reward for finishing puzzle")
            return 500.
        elif push_on_goal:
            #next_state = self.next_state(state, action, sokoban_map)
            #self.inspiration.append((copy.deepcopy(state), copy.deepcopy(self.environment.has_scored)))
            return 50. 
        elif state_hash in self.environment.deadlock_table and any([self.environment.deadlock_table[state_hash][key] for key in self.environment.deadlock_table[state_hash]]):
            #print('deadlock reward')
            return -5.
        # elif box_pushing:
        #   return -0.5
        # elif self.environment.is_deadlock():
        #   #print("deadlock reward")
        #   return -2
        else:
            return -1.


    def target(self, state, action):
        next_state = self.environment.next_state(state, action)
        if self.environment.is_goal_state(next_state):
            return self.reward(state, action)

        #qmax = 
        return self.reward(state, action) + self.discount_factor*torch.max(self.predict(next_state))


    def argaction(self, action):
        for index, value in enumerate(self.actions):
            if action == value:
                return index


    def predict(self, state):
        tensor_state = torch.from_numpy(state).view(1, 2, 9, 9).float()
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor_state)


    def episode(self, draw = False, evaluate = False, max_iterations=5000):

        state = np.copy(self.environment.state)

        num_iterations = 0

        if draw:
            self.environment.draw(state)
        while not self.environment.is_goal_state(state) and not self.environment.is_deadlock(state) and num_iterations < max_iterations:
            tensor_state = torch.from_numpy(state).view(1, 2, 9, 9).float()

            
            if not evaluate:
                if len(self.replay_buffer) >= self.minibatch_size:
                    self.model.train()

                    samples = self.replay_buffer.sample(self.minibatch_size)
                    
                    batch = np.stack(samples, axis=0)

                    tensor_state = torch.from_numpy(batch).view(-1, 2, 9, 9).float()
                    #print(tensor_state.size())

                    y_pred = self.model(tensor_state)
                    y = torch.tensor([[self.target(state, action) for action in self.actions] for state in samples])
                    self.model.train()
                    
                    loss = self.criterion(y_pred, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                chosen_action = random.choice(self.actions)
            else:
                qvalues = self.predict(state)
                if self.verbose:
                    print(qvalues)
                chosen_action = self.actions[torch.argmax(qvalues)]


            state = self.environment.next_state(state, chosen_action)

            self.replay_buffer.add(np.copy(state))

            
            if draw:
                self.environment.draw(state)

            

        goal_flag = self.environment.is_goal_state(state)

        return goal_flag, num_iterations


    def save(self, filename):
        '''
        Saves the agent's weights and training progress.
        '''

        #filepath = Path(filename)

        torch.save(self.model.state_dict(), filename)


    def load(self, filename):
        '''
        Loads the agent's weight sand training progress.
        '''
        filepath = Path(filename)

        if not filepath.exists():
            raise ValueError("Path does not exist.")
        if not filepath.is_file():
            raise ValueError("Path is not a valid file.")

        self.model.load_state_dict(torch.load(filename))

