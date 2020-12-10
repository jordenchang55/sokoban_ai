

from agent import Agent
import numpy as np
import random
from queue import PriorityQueue
from functools import total_ordering
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@total_ordering
class Node:
    def __init__(self, f, neighbor, path):
        self.data = (f, neighbor, path)

    def __repr__(self):
        print(f"Node(f={data[0]}, neighbor={data[1]}, path={data[2]})")

    def __lt__(self, other):
        return self.data[0] < other.data[0]
    def __eq__(self, other):
        return self.data[0] == other.data[0]

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        return next(self.data)

class BoxAgent(Agent):
    def __init__(self, environment, discount_factor = 0.95, replay_rate = 0.2, verbose = False,*args, **kwargs):
        #super()
        super().__init__(environment, args, kwargs)

        self.print_threshold = 500

        #memoization tables
        self.q_table = {}
        self.path_table = {}

        self.discount_factor = discount_factor

        self.verbose = verbose

        self.num_episodes = 0

        self.q_sequence = []

        #internal reward metric
        self.boxes_scored = 0

        self.greedy_hash = self.encode(self.environment.state, self.get_actions(self.environment.state)[0])


    def encode(self, state, action):
        return (state[1,:,:].tobytes(), action.tobytes())

    def reward(self, state, action):
        box, box_action = action

        next_box_position = box + box_action

        push_on_goal = (tuple(next_box_position) in self.environment.storage)
        #print(f"push_on_goal:{push_on_goal}, num_scored:{self.environment.count_boxes_scored(self.next_state(state, action))}")

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
            #print("super goal")
            return 100.#500.
        elif push_on_goal and self.boxes_scored < self.environment.count_boxes_scored(self.next_state(state, action)):
            #print("goal")
            return 10.#. 
        elif state_hash in self.environment.deadlock_table and any([self.environment.deadlock_table[state_hash][key] for key in self.environment.deadlock_table[state_hash]]):
            #print('deadlock reward')
            return -10.
        # elif box_pushing:
        #   return -0.5
        # elif self.environment.is_deadlock():
        #   #print("deadlock reward")
        #   return -2
        else:
            return -1.

    def path_encoding(self, state, posb):
        return (state.tobytes(), posb.tobytes())


    

    def astar(self, state, destination):
        pqueue = PriorityQueue()
        
        # self.environment.draw(state)

        # ax = plt.gca()
        # rect = patches.Rectangle((destination[0]+0.5, destination[1]+0.5),-1,-1,linewidth=0.5,edgecolor='red',facecolor='red')
        # ax.add_patch(rect)
        # plt.show(block=False)
        # plt.pause(0.05)

        start = np.array(self.environment.get_player(state))

        pqueue.put(Node(np.linalg.norm(start - destination), start, []))

        visited = set([start.tobytes()])
        while not pqueue.empty():

            # for item in pqueue.queue:
            #     rect = patches.Rectangle((item.data[1][0]+0.5, item.data[1][1]+0.5),-1,-1,linewidth=0.5,edgecolor='green',facecolor='green')
            #     ax.add_patch(rect)
            # plt.show(block=False)
            # plt.pause(0.05)

            path_cost, position, prev_path = pqueue.get()


            if (position == destination).all():
                ##reached goal
                break

            for action in self.actions:
                neighbor = position + action
                x, y = neighbor
                if neighbor.tobytes() not in visited and state[1, x, y] == 0 and state[2, x, y] == 0:
                    ##if empty
                    visited.add(neighbor.tobytes())

                    f = path_cost - np.linalg.norm(position-destination) + np.linalg.norm(neighbor - destination) + 1. #edge cost is 1

                    pqueue.put(Node(f, neighbor, prev_path[:] + [action]))


        if pqueue.empty():
            return None


        return prev_path[:]






    def find_path(self, state, next_location):

        #player = self.environment.get_player(state)

        path_hash = self.path_encoding(state, next_location)
        if path_hash in self.path_table:
            return self.path_table[path_hash]

        path = self.astar(state, next_location)

        self.path_table[self.path_encoding(state, next_location)] = path

        return path


    def get_actions(self, state):
        self.environment.draw(state)
        ax = plt.gca()
        possible_actions = []
        for i in range(self.environment.xlim+1):
            for j in range(self.environment.ylim+1):
            #print(box)
                if state[1, i, j] == 1:
                    box = np.array([i,j])
                    for action in self.actions:
                        after = box + action
                        before = box - action

                        

                        #print(f"before:{before}")
                        #print(self.find_path(state, before))
                        if state[1, before[0], before[1]] == 0 and state[2, before[0], before[1]] == 0 and state[1, after[0], after[1]] == 0 and state[2, after[0], after[1]] == 0 and not self.environment.is_deadlock(state) and self.find_path(state, before):
                            #if empty and reachable
                            possible_actions.append(np.array((box, action))) #actions are box neighbor pairs
                            rect = patches.Rectangle((after[0]+0.5, after[1]+0.5),-1,-1,linewidth=0.5,edgecolor='green',facecolor='green')
                            ax.add_patch(rect)
                            plt.show(block=False)
                            plt.pause(0.1)
        # print(f"possible:{len(possible_actions)}")
        # if len(possible_actions) == 0:
        #     self.environment.pause = 10.
        #     self.environment.draw(state)
        return possible_actions


    def get_greedy_rate(self):
        if self.greedy_hash in self.q_table:
            q_value = self.q_table[self.greedy_hash]
            #print(q_value)


            rate = ((q_value*self.discount_factor)/500)
            #print(rate)

            if rate > 0.9:
                return 0.9
            else:
                return rate
        else:
            return 0
        #return 0.3


    def next_state(self, state, action):
        '''
        Defines the next state in the agent's state space. Different from the environment state.
        '''

        box, box_action = action

        next_state = np.copy(state)
        next_position = box + box_action

        
        next_state[1, box[0], box[1]] = 0
        next_state[1, next_position[0], next_position[1]] = 1

        return next_state



    def update(self, state, action):
        #print(action)
        if self.encode(state, action) not in self.q_table:
            self.q_table[self.encode(state, action)] = 0. 

        next_state = self.next_state(state, action)

        next_actions = self.get_actions(next_state)
        for possible_action in next_actions:
            if self.encode(next_state, possible_action) not in self.q_table:
                self.q_table[self.encode(next_state, possible_action)] = 0.

        if next_actions:
            qmax = np.amax(np.array([self.q_table[self.encode(next_state, possible_action)] for possible_action in next_actions]))
        else:
            qmax = -1.
        self.q_table[self.encode(state, action)] += (self.reward(state, action) + self.discount_factor*qmax - self.q_table[self.encode(state, action)])


    def learn(self, state):

        if random.random() >= self.get_greedy_rate():
            #print("random")
            actions = self.get_actions(state)
            if actions:
                action = random.choice(actions)
            else:
                action = None
        else:
            action = self.evaluate(state)

        if action is not None:
            self.update(state, action)
        return action




    def evaluate(self, state):
        chosen_action = None
        chosen_value = 0.


        for possible_action in self.get_actions(state):

            if self.verbose:
                print(f"{environment.direction_to_str(possible_action[1])}:{self.q_table[self.encode(state, possible_action)]}")
            
            if self.encode(state, possible_action) not in self.q_table:
                self.q_table[self.encode(state, possible_action)] = 0. #represents an unseen state... not ideal while evaluating

            # print(possible_action)
            # print(self.qtable[self.encode(state, possible_action)])   

            if chosen_action is None:
                chosen_action = possible_action
                chosen_value = self.q_table[self.encode(state, possible_action)]
            else:
                potential_value = self.q_table[self.encode(state, possible_action)]
                if chosen_value < potential_value:
                    #keep this one
                    chosen_action = possible_action
                    chosen_value = potential_value

        self.q_sequence.append(chosen_value)
        return chosen_action

    def episode(self, draw=False, evaluate=False, max_iterations=8000):
        action_sequence = []
        self.q_sequence = []
        self.num_episodes += 1

        self.boxes_scored = 0

        self.environment.reset()
        state = self.environment.state

        num_iterations = 0 

        print(f"{self.num_episodes:5d}.{0:7d}:")
        while not self.environment.is_goal_state(state) and not self.environment.is_deadlock(state) and num_iterations < max_iterations:


            if not evaluate:
                box_action = self.learn(state)
            else:
                box_action = self.evaluate(state)
        
            if box_action is None:
                break
            box, action = box_action            

            next_player_location = box - action

            path = self.find_path(state, next_player_location)

            if path:
                for path_action in path:
                    state = self.environment.next_state(state, path_action)
                    # if draw:
                    #     self.environment.draw(state)

            num_boxes_scored = self.environment.count_boxes_scored(state)
            if self.boxes_scored < num_boxes_scored:
                self.boxes_scored = num_boxes_scored

            state = self.environment.next_state(state, action)
            if draw:
                self.environment.draw(state)

            if num_iterations > 0 and num_iterations%self.print_threshold == 0:
                print(f"     .{num_iterations:7d}:")

            num_iterations += 1

        if self.environment.is_deadlock(state):
            for action in self.get_actions(state):
                self.q_table[self.encode(state, action)] = self.reward(state, action)

        if num_iterations%self.print_threshold != 0:
            print(f"     .{num_iterations:7d}:")

        goal_flag = self.environment.is_goal_state(state)

        if evaluate:
            print("-"*20)
            print(f"evaluation :{goal_flag}")
            if self.q_sequence:
                print(f"mean q(s,a):{np.array(self.q_sequence).mean():.4f}")
            if goal_flag:
                print(f"iterations :{num_iterations}")
            print(f"greedy_rate:{self.get_greedy_rate()}")
            print("-"*20)



        return goal_flag, num_iterations#, action_sequence