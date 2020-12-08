import torch
import torch.nn as nn
import torch.nn.functional as funct
from agent import Agent


class SokobanNet(nn.Module):
    def __init__(self, xlim, ylim):
        super(SokobanNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 3, 3, stride=1)
        self.conv4 = nn.Conv2d(3, 3, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(3)
        self.bn4 = nn.BatchNorm2d(3)

        self.fc1 = nn.Linear(3*xlim*ylim, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, 4)

    def forward(self, s):
        #s = s.view()

        s = funct.relu(self.bn1(self.conv1(s)))
        s = funct.relu(self.bn2(self.conv2(s)))
        s = funct.relu(self.bn3(self.conv3(s)))
        s = funct.relu(self.bn4(self.conv4(s)))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        q = self.fc4(s)

        return q



class DeepQAgent(Agent):



    def __init__(self, environment, learning_rate=1.0, discount_factor=0.95, verbose=False):
        super().__init__(self, environment)

        # algorithm variable


        #if torch.cuda.is_available():
        self.model = SokobanNet(10, 10)
        if torch.cuda.is_available():
            self.model.cuda()

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6)


    def reward(self, state, action):
        box_position = np.array(self.environment.get_player()) + action
        next_box_position = box_position + action
        
        box_pushing = state[box_position[0], box_position[1], 1] == 1. and state[next_box_position[0], next_box_position[1], 1] == 0.
        push_on_goal = box_pushing and (tuple(next_box_position) in self.environment.storage)

        # not_scored = False
        # for i in range(len(self.environment.state[2:])):
        #   if (box_position == self.environment.state[2+i]).all():
        #       not_scored = not self.environment.has_scored[i]



        goal_reach = all([state[place[0], place[1], 1] == 1 for place in self.environment.storage])
        if push_on_goal:
            goal_reach = True
            set_difference = self.environment.storage.difference({tuple(next_box_position)})
            for place in set_difference:
                if state[place[0], place[1], 1] == 0:
                    goal_reach = False
        else:
            goal_reach = False

        #boxes_hash = self.next_state(state, action, )[2:].tobytes()
        if goal_reach:
            #print("reward for finishing puzzle")
            return 500.
        elif push_on_goal:
            #next_state = self.next_state(state, action, sokoban_map)
            #self.inspiration.append((copy.deepcopy(state), copy.deepcopy(self.environment.has_scored)))
            if len(self.inspiration) > 10: 
                self.inspiration.pop(0)
            return 50. 
        elif boxes_hash in self.environment.deadlock_table and any([self.environment.deadlock_table[boxes_hash][key] for key in self.environment.deadlock_table[boxes_hash]]):
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
        if self.environment.is_goal_state(self.next_state(state, action)):
            return self.reward(state, action)

        #qmax = 
        return self.reward + self.discount_factor*torch.max(self.model(self.next_state(state, action)))


    def argaction(self, action):
        for index, value in enumerate(self.actions):
            if action == value:
                return index


    def episode(self, evaluate = False):


        while not self.environment.is_goal_state(state) and not self.environment.is_deadlock(state):

            y_pred = self.model(state)

            if not evaluate:

                loss = self.criterion(y_pred, target(state, action))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                action = random.choice(self.actions)
            else:
                action = self.actions[torch.argmax(y_pred)]

