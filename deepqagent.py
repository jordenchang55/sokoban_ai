import csv
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.nn.init as init

from agent import Agent


class SokobanNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(SokobanNet, self).__init__()
        self.dropout = dropout

        channels = 4
        self.conv1 = nn.Conv2d(4, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)

        self.fc1 = nn.Linear((4 * (DeepQAgent.INPUT_SIZE - 4) ** 2), 256)  ##CHANGE TO 256 128
        self.fc_bn1 = nn.BatchNorm1d(256)
        init.kaiming_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(256, 128)
        init.kaiming_normal_(self.fc2.weight)
        self.fc_bn2 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 4)

    def forward(self, s):
        # s = s.view()

        s = funct.relu(self.bn1(self.conv1(s)))
        # print(s.size())
        s = funct.relu(self.bn2(self.conv2(s)))
        s = funct.relu(self.bn3(self.conv3(s)))
        s = funct.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, 4 * ((DeepQAgent.INPUT_SIZE - 4) ** 2))

        assert torch.isnan(s).any() == False, print("NaN numbers in tensor.")

        s = funct.dropout(funct.relu(self.fc_bn1(self.fc1(s))), p=self.dropout,
                          training=self.training)  # batch_size x 1024
        s = funct.dropout(funct.relu(self.fc_bn2(self.fc2(s))), p=self.dropout,
                          training=self.training)  # batch_size x 512

        assert torch.isnan(s).any() == False, print("NaN numbers in tensor.")

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
    INPUT_SIZE = 15

    def __init__(self, environment, learning_rate=1e-6, discount_factor=0.8, minibatch_size=32, buffer_size=5000000,
                 quiet=False, verbose=False):
        super().__init__(environment, quiet, verbose)

        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.buffer_size = buffer_size

        if self.environment.xlim > DeepQAgent.INPUT_SIZE - 1 or self.environment.ylim > DeepQAgent.INPUT_SIZE - 1:
            raise ValueError("Map size too large for current DeepQAgent implementation.")
        else:
            self.pad_config = [(0, DeepQAgent.INPUT_SIZE - (self.environment.xlim + 1)),
                               (0, DeepQAgent.INPUT_SIZE - (self.environment.ylim + 1))]

        # if torch.cuda.is_available():
        self.model = SokobanNet()
        if torch.cuda.is_available():
            self.model.cuda()
            self.cuda_device = torch.device('cuda')
        else:
            self.cuda_device = None

        # mse and sgd algorithm
        self.criterion = nn.MSELoss(reduction='sum').cuda() if self.cuda_device else nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # replay buffer. new per agent
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)

        self.action_sequence = None

        # for profiling training and episodes
        self.training_times = []
        self.episode_times = []

        # for graphing loss
        self.running_loss = 0
        self.losses = []

        # internal record of number episodes run
        self.times_trained = 0
        self.print_threshold = 50

        self.boxes_scored = 0

    def get_greedy_rate(self):
        if self.num_episodes < 1000000:
            return 0.8 * self.num_episodes / (1000000)
        return 0.8

    def load_environment(self, environment):
        self.environment = environment
        self.pad_config = [(0, DeepQAgent.INPUT_SIZE - (self.environment.xlim + 1)),
                           (0, DeepQAgent.INPUT_SIZE - (self.environment.ylim + 1))]

    def reward(self, state, action):
        box_position = np.array(self.environment.get_player(state)) + action
        next_box_position = box_position + action

        box_pushing = state[1, box_position[0], box_position[1]] == 1. and state[
            1, next_box_position[0], next_box_position[1]] == 0.
        push_on_goal = box_pushing and (tuple(next_box_position) in self.environment.storage)

        # not_scored = False
        # for i in range(len(self.environment.state[2:])):
        #   if (box_position == self.environment.state[2+i]).all():
        #       not_scored = not self.environment.has_scored[i]
        next_state = self.environment.next_state(state, action)

        state_hash = state.tobytes()
        if self.environment.is_goal_state(next_state):
            return 1.  # 500.
        elif self.environment.count_boxes_scored(state) < self.environment.count_boxes_scored(next_state):
            #print("reward on goal")
            return 1.  # .
        elif self.environment.count_boxes_scored(state) > self.environment.count_boxes_scored(next_state):
            #print("reward off")
            return -1.
        elif self.environment.is_deadlock(state):
            return -1.
        else:
            return 0

    def target(self, state, action):
        next_state = self.environment.next_state(state, action)
        if self.environment.is_goal_state(next_state):
            return self.reward(state, action)

        # qmax =
        return self.reward(state, action) + self.discount_factor * torch.max(self.predict(next_state))

    def argaction(self, action):
        for index, value in enumerate(self.actions):
            if action == value:
                return index

    def train(self):
        training_start = time.process_time()

        self.model.train()

        samples = self.replay_buffer.sample(self.minibatch_size)

        states, targets = zip(*samples)
        # print(len(states))
        batch = np.stack(states, axis=0)

        tensor_state = torch.from_numpy(batch).view(-1, 4, DeepQAgent.INPUT_SIZE, DeepQAgent.INPUT_SIZE).float()
        if self.cuda_device:
            tensor_state = tensor_state.contiguous().cuda()
        # print(tensor_state.size())

        y_pred = self.model(tensor_state)
        if self.cuda_device:
            y = torch.tensor(targets, device=self.cuda_device)
        else:
            y = torch.tensor(targets)
        # y = torch.tanh(y)
        # self.verbose_print(f"{y_pred}, {y}")
        self.model.train()

        loss = self.criterion(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.episode_print(f"{loss.item():.4f}")
        self.running_loss += loss.item()
        self.times_trained += 1

        self.training_times[-1].append(time.process_time() - training_start)

    def predict(self, state):
        pad_state = np.pad(state, [(0, 0), *self.pad_config])
        tensor_state = torch.from_numpy(pad_state).view(1, 4, DeepQAgent.INPUT_SIZE, DeepQAgent.INPUT_SIZE).float()
        if self.cuda_device:
            tensor_state = tensor_state.contiguous().cuda()
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor_state)

    def episode(self, draw=False, evaluate=False, max_iterations=5000):
        episode_start = time.process_time()
        self.num_episodes += 1

        state = np.copy(self.environment.state)
        self.training_times.append([])
        self.num_iterations = 0

        self.action_sequence = []
        self.q_sequence = []

        self.running_loss = 0
        self.times_trained = 0

        self.episode_print()

        if draw:
            self.environment.draw(state)
        while not self.environment.is_goal_state(state) and not self.environment.is_deadlock(
                state) and self.num_iterations < max_iterations:
            # tensor_state = torch.from_numpy(state).view(1, 4, 9, 9).float()

            if not evaluate:
                if len(self.replay_buffer) >= self.minibatch_size:
                    self.train()

                if random.random() > self.get_greedy_rate():
                    chosen_action = random.choice(self.actions)
                else:
                    chosen_action = self.actions[torch.argmax(self.predict(state))]
            else:
                qvalues = self.predict(state)
                # if self.verbose:
                # print(f"{qvalues}:")
                self.q_sequence.append(np.array(torch.max(qvalues).cpu()))
                chosen_action = self.actions[torch.argmax(qvalues)]
            # print(f"     .{self.num_iterations:7d}:{qvalues},{chosen_action}")

            self.action_sequence.append(chosen_action)
            state = self.environment.next_state(state, chosen_action)

            # [UP RIGHT DOWN LEFT], [RIGHT DOWN LEFT UP], [DOWN, LEFT UP RIGHT] LEFT UP RIGHT DOWN]
            rotate = [np.pad(state, [(0, 0), *self.pad_config])]
            for k in range(1, 4):
                rotate.append(np.rot90(rotate[0], k, (1, 2)))
            targets = [self.target(state, action) for action in self.actions]
            self.replay_buffer.add((rotate[0], targets))
            targets.append(targets.pop(0))
            self.replay_buffer.add((rotate[1], targets))
            targets.append(targets.pop(0))

            self.replay_buffer.add((rotate[2], targets))
            targets.append(targets.pop(0))
            self.replay_buffer.add((rotate[3], targets))  ##all symmetries

            num_boxes_scored = self.environment.count_boxes_scored(state)
            if self.boxes_scored < num_boxes_scored:
                self.boxes_scored = num_boxes_scored

            if draw:
                self.environment.draw(state)

            if self.num_iterations > 0 and self.num_iterations % self.print_threshold == 0:
                self.episode_print()

            self.num_iterations += 1

        # if self.num_iterations%self.print_threshold != 0:
        #     self.episode_print()
        goal_flag = self.environment.is_goal_state(state)

        if evaluate:
            qvalues = np.array(self.q_sequence)
            qmean = qvalues.mean()
            print("-" * 20)
            print(f"evaluation :{goal_flag}")
            print(f"mean q(s,a):{qmean:.3f}")
            if goal_flag:
                print(f"iterations :{self.num_iterations}")
            print("-" * 20)

        if self.times_trained != 0:
            self.losses.append(self.running_loss / self.times_trained)
            self.episode_print(f"loss:{self.running_loss / self.times_trained:.3f}")

        self.episode_times.append((self.num_iterations, time.process_time() - episode_start))

        return goal_flag, self.num_iterations, self.action_sequence

    def save_sequence(self, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for action in self.action_sequence:
                writer.writerow(action)

    def save(self, filename):
        '''
        Saves the agent's weights and training progress.
        '''

        # filepath = Path(filename)

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
