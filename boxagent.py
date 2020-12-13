

from agent import Agent
import numpy as np
import random
from queue import PriorityQueue
from functools import total_ordering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stateenvironment import State
import copy
from collections import deque
from time import process_time
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
    def __init__(self, environment, discount_factor = 1.0, replay_rate = 0.2, quiet = False, verbose = False):
        #super()
        super().__init__(environment, quiet, verbose)

        self.print_threshold = 500

        #memoization tables
        self.q_table = {}
        self.path_table = {}

        self.discount_factor = discount_factor

        self.draw = False

        self.q_sequence = []

        #internal reward metric
        self.boxes_scored = 0


        #self.environment.draw(self.environment.state)
        #self.greedy_hash = self.encode(self.environment.state, self.get_actions(self.environment.state)[0])



    def encode(self, state, action):
        return (state.boxes.tobytes(), state.max_score, action.tobytes())

    def reward(self, state, action):
        box, box_action = action

        next_box_position = box + box_action

        push_on_goal = (tuple(next_box_position) in state.storage)
    

        next_state = self.next_state(state, action)
        # if push_on_goal:
        #     print(next_state.max_score)
        if self.environment.is_goal_state(next_state):
            return 500.
        elif push_on_goal and state.max_score < next_state.max_score:
            #print("reward")
            return 50
        elif self.environment.is_deadlock(state):
            return -2.
        else:
            return -1.

    def path_encoding(self, state, posb):
        return (state.map.tobytes(), posb.tobytes())


    

    def search(self, state, destination):
        pqueue = PriorityQueue()
        
        if self.verbose and self.draw:
            self.environment.draw(state)

            ax = plt.gca()
            rect = patches.Rectangle((destination[0]+0.5, destination[1]+0.5),-1,-1,linewidth=0.5,edgecolor='firebrick',facecolor='firebrick')
            ax.add_patch(rect)
            plt.show(block=False)
            plt.pause(0.01)

        start = np.array(self.environment.get_player(state))

        pqueue.put(Node(np.linalg.norm(start - destination), start, []))

        visited = set([start.tobytes()])
        while not pqueue.empty():
            if self.verbose and self.draw:
                for item in pqueue.queue:
                    rect = patches.Rectangle((item.data[1][0]+0.5, item.data[1][1]+0.5),-1,-1,linewidth=0.5,edgecolor='green',facecolor='green')
                    ax.add_patch(rect)
                plt.show(block=False)
                plt.pause(0.01)

            cost, position, prev_path = pqueue.get()


            if (position == destination).all():
                #print("goal!")
                ##reached goal
                break

            for action in self.actions:
                neighbor = position + action
                if neighbor.tobytes() not in visited and state.map[tuple(neighbor)] == State.EMPTY:
                    ##if empty
                    visited.add(neighbor.tobytes())

                    #f = cost - np.linalg.norm(position-destination) + np.linalg.norm(neighbor - destination) + 1. #edge cost is 1
                    
                    h = np.linalg.norm(neighbor - destination) #dfs, slightly faster for euclidian spaces
                    pqueue.put(Node(h, neighbor, prev_path[:] + [action]))


        if (position == destination).all():
            return prev_path[:]
            
        return None

    def find_path(self, state, next_location):

        #player = self.environment.get_player(state)

        path_hash = self.path_encoding(state, next_location)
        if path_hash in self.path_table:
            return self.path_table[path_hash]

        path = self.search(state, next_location)

        self.path_table[self.path_encoding(state, next_location)] = path

        return path


    def get_actions(self, state):
        if self.verbose and self.draw:
            self.environment.draw(state)


        ax = plt.gca()
        possible_actions = []
        for box in state.boxes:
            #print(box)
            for action in self.actions:
                after = box + action
                before = box - action

                is_empty = state.map[tuple(before)] <= State.PLAYER and state.map[tuple(after)] <= State.PLAYER #cheat to say empty or player
                
                if is_empty:
                    next_state = self.next_state(state, np.array([box, action]))
                else:
                    next_state = None
                # path =  if is_empty else None
                # is_deadlock = 
                # if self.draw:
                #     print(f"{is_empty}, {path}, {is_deadlock}")
                if is_empty and not self.environment.is_deadlock(next_state) and self.find_path(state, before) is not None:
                    #if empty and reachable
                    possible_actions.append(np.array((box, action))) #actions are box neighbor pairs
                    if self.verbose and self.draw:
                        rect = patches.Rectangle((before[0]+0.5, before[1]+0.5),-1,-1,linewidth=0.5,edgecolor='darkorange',facecolor='darkorange')
                        ax.add_patch(rect)

        if self.verbose and self.draw:
            plt.show(block=False)
            plt.pause(0.5)
        # # print(f"possible:{len(possible_actions)}")
        # # if len(possible_actions) == 0:
        # #     self.environment.pause = 10.
        # #     self.environment.draw(state)
        return possible_actions


    def get_greedy_rate(self):
        # if self.greedy_hash in self.q_table:
        #     q_value = self.q_table[self.greedy_hash]
        #     #print(q_value)


        #     rate = ((q_value*self.discount_factor)/500)
        #     #print(rate)

        #     if rate > 0.9:
        #         return 0.9
        #     else:
        #         return rate
        # else:
        #     return 0

        if self.num_episodes > 50000:
            return 0.8
        else:
            return 0.8*self.num_episodes/50000
        #return 0.3


    def next_state(self, state, action):
        '''
        Defines the next state in the agent's state space. Different from the environment state.
        '''

        box, box_action = action

        next_state = copy.deepcopy(state)
        next_position = box + box_action

        if state.map[tuple(next_position)] > State.PLAYER:
            self.environment.draw(state)
            print(next_position)
            plt.show(block=True)
            assert state.map[tuple(next_position)] <= State.PLAYER, "place should be empty"
        next_state.map[tuple(box)] = State.EMPTY
        next_state.map[tuple(next_position)] = State.BOX

        for index in range(len(state.boxes)):
            if (next_state.boxes[index] == box).all():
                next_state.boxes[index] = next_position
                break

        if tuple(next_position) in next_state.storage:
            score = self.environment.count_boxes_scored(next_state)
            if next_state.max_score < score:
                next_state.max_score = score


        # for box in next_state.boxes:
        #     assert next_state.map[tuple(box)] == State.BOX, "boxes misaligned with map."
        return next_state



    def update(self, state, action):
        #print(action)
        if self.encode(state, action) not in self.q_table:
            self.q_table[self.encode(state, action)] = 1. 

        next_state = self.next_state(state, action)

        if not self.environment.is_goal_state(next_state):
            next_actions = self.get_actions(next_state)
            for possible_action in next_actions:
                if self.encode(next_state, possible_action) not in self.q_table:
                    self.q_table[self.encode(next_state, possible_action)] = 1.

            if next_actions:
                qmax = np.amax(np.array([self.q_table[self.encode(next_state, possible_action)] for possible_action in next_actions]))
            else:
                qmax = -2.
            self.q_table[self.encode(state, action)] = (self.reward(state, action) + self.discount_factor*qmax )
        elif self.environment.is_deadlock(next_state):
            self.q_table[self.encode(state, action)] = self.reward(state, action)
        else:
            self.q_table[self.encode(state, action)] = self.reward(state, action)


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

            if self.encode(state, possible_action) not in self.q_table:
                self.q_table[self.encode(state, possible_action)] = 1. #represents an unseen state... not ideal while evaluating


            if self.verbose and self.draw:
                #print(possible_action)
                next_state = self.next_state(state, possible_action)
                #print(f"{possible_action[0]}, {self.environment.direction_to_str(possible_action[1])}:{self.q_table[self.encode(state, possible_action)]}:{self.environment.is_deadlock(next_state)}")
            
            next_state = self.next_state(state, possible_action)

            self.episode_print(f"{self.environment.direction_to_str(possible_action[1])}={self.q_table[self.encode(state, possible_action)]:.4f}, reward={self.reward(state, possible_action)}, goal={self.environment.is_goal_state(next_state)}, box_count={self.environment.count_boxes_scored(next_state)}")

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

    def replay(self, action_sequence, update=True):
        self.environment.reset()
        self.boxes_scored = 0

        state = copy.deepcopy(self.environment.state)
        for box_action in action_sequence:
            if self.draw:
                self.environment.draw(state)
            if update:
                self.update(state, box_action)
            #print(f"q_table:{self.q_table[self.encode(state, box_action)]}")
            box, action = box_action
            next_player_location = box - action

            path = self.find_path(state, next_player_location)

            if path:
                for path_action in path:
                    state = self.environment.next_state(state, path_action)
            state = self.environment.next_state(state, action)

        if self.draw:
            self.environment.draw(state)



    def episode(self, draw=False, evaluate=False, max_iterations=8000):
        box_action_sequence = []
        action_seqeuence = []
        self.q_sequence = []
        self.num_episodes += 1
        self.num_iterations = 0 
        self.boxes_scored = 0

        if self.num_episodes == 1:
            self.start_time = process_time()

        self.environment.reset()
        state = copy.deepcopy(self.environment.state)
        self.draw = draw

        

        if draw:
            self.environment.draw(state)
        self.episode_print()

        previous = deque(maxlen=4)
        def count(previous, current_hash):
            count = 0
            for item in previous:
                if item == current_hash:
                    count += 1
            return count

        while not self.environment.is_goal_state(state) and not self.environment.is_deadlock(state) and self.num_iterations < max_iterations:


            if not evaluate:
                box_action = self.learn(state)
            else:
                box_action = self.evaluate(state)
        
            if box_action is None:
                break
            box, action = box_action  

            current_hash = self.encode(state, box_action)  
            if count(previous, current_hash) >= 2:
                break
            else:
                previous.append(current_hash)        

            next_player_location = box - action

            path = self.find_path(state, next_player_location)

            if path:
                for path_action in path:
                    state = self.environment.next_state(state, path_action)
                    action_sequence.append(path_action)

            state = self.environment.next_state(state, action)
            action.sequence.append(action)

            num_boxes_scored = self.environment.count_boxes_scored(state)
            if self.boxes_scored < num_boxes_scored:
                self.boxes_scored = num_boxes_scored
                if not self.environment.is_goal_state(state):
                    self.episode_print("Box pushed on goal.")
                else:
                    self.episode_print("Goal state reached.")

            if self.num_iterations > 0 and self.num_iterations%self.print_threshold == 0:
                self.episode_print()
                
            if draw:
                self.environment.draw(state)

            box_action_sequence.append(box_action)
            self.num_iterations += 1

        goal_flag = self.environment.is_goal_state(state)

        if not evaluate and self.environment.count_boxes_scored(state) > 0:
            self.replay(action_sequence)
        if self.num_iterations%self.print_threshold != 0:
            self.episode_print(f"goal={goal_flag}")

        

        if evaluate:
            qmean = np.array(self.q_sequence).mean()
            self.standard_print("-"*20)
            self.standard_print(f"goal reached:{goal_flag}")
            if self.q_sequence:
                self.standard_print(f"mean q(s,a) :{qmean:.4f}")
            if goal_flag:
                self.standard_print(f"iterations  :{self.num_iterations}")
            self.standard_print(f"greedy rate :{self.get_greedy_rate():.4f}")
            self.standard_print(f"time taken  :{process_time()-self.start_time}")
            self.standard_print("-"*20)

            if qmean > 500:
                self.draw = True
                self.replay(box_action_sequence,    update = False)

        self.draw = False

        return goal_flag, self.num_iterations, action_sequence