
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from environment import Environment
class DeepEnvironment(Environment):


    def __init__(self, filename, walls, boxes, player, storage, xlim, ylim, pause=0.05):
        super().__init__(filename, xlim, ylim)
        

        #0 player plane, 1 box plane, 2 wall plane, 3 storage plane
        self.state = np.zeros((4, xlim+1, ylim+1), dtype=np.double)#torch.zeros(xlim+1, ylim+1, 2)
        #self.walls = np.zeros((xlim+1, ylim+1)) #torch.zeros(xlim+1, ylim+1)
        self.xlim = xlim
        self.ylim = ylim

        self.boxes = np.array(boxes)
        self.num_boxes = len(boxes)
        #print(self.boxes)

        self.storage = set(storage)

        self.pause = pause


        for wall in walls:
            #print(wall)
            #self.walls[wall[0], wall[1]] = 1.
            self.state[2, wall[0], wall[1]] = 1.
        for box in boxes:
            self.state[1, box[0], box[1]] = 1.
        for place in storage:
            self.state[3, place[0], place[1]] = 1.
        self.state[0, player[0], player[1]] = 1.

        self.deadlock_table = {}

        self.original_state = copy.deepcopy(self.state)
        self.original_boxes = copy.deepcopy(self.boxes)
        
        
        self.state_hash = None

        self.cache_miss = 0
        self.cache_hit = 0



#     def save_state(self):
#         self.saved_map = copy.deepcopy(self.map)
#         self.saved_state = copy.deepcopy(self.state)
#         self.saved_scores = copy.deepcopy(self.has_scored)

#     def reset_to_save(self):
#         if self.saved_map is None:
#             print("NEED TO SAVE BEFORE RESET!")
#         else:
#             self.map = copy.deepcopy(self.saved_map)
#             self.state = copy.deepcopy(self.saved_state)
#             self.has_scored = copy.deepcopy(self.saved_scores)

    def reset(self):
        # print("reset!")
        # print(f"player:{self.state[0]}")
        # print(f"reset_player:{self.original_player}")
        self.state = copy.deepcopy(self.original_state)
        self.boxes = copy.deepcopy(self.original_boxes)

#     def reset_map(self):
#         for i in range(self.xlim):
#             for j in range(self.ylim):
#                 if self.map[i, j] == BOX:
#                     self.map[i, j] = EMPTY
        
#         self.map[tuple(self.state[0])] = PLAYER
#         for box in self.state[2:]:
#             self.map[tuple(box)] = BOX


#     def is_goal(self):
#         for place in self.storage:
#             if self.map[place] != BOX:
#                 return False

#         return True

    def is_valid(self, location):
        x, y = location

        return (x >= 0 and x <= self.xlim and y >= 0 and y <= self.ylim)

    def count_boxes_scored(self, state):
        count = 0
        for place in self.storage:
            if state[1, place[0], place[1]] == 1:
                count += 1
        return count

    def is_goal_state(self, state):
        for place in self.storage:
            if state[1, place[0], place[1]] != 1:
                return False

        return True

    def get_player(self, state):
        return np.unravel_index(np.argmax(state[0, :, :]), state[0, :, :].shape)
    def get_neighbors(self, location):
        return [location + direction for direction in Environment.DIRECTIONS]


    def is_frozen(self, state, location, previous=None):

        if location.tobytes() in self.deadlock_table[self.state_hash]:
            return self.deadlock_table[self.state_hash][location.tobytes()]

        # if not previous:
        #   previous = set([])
        neighbors = self.get_neighbors(location)
        previous.add(tuple(location))
        if tuple(location) not in self.storage:
            for i in range(len(neighbors)):
                neighbor = tuple(neighbors[i])
                next_neighbor = tuple(neighbors[(i+1)%len(neighbors)])

                if state[2, neighbor[0], neighbor[1]] == 1 and state[2, next_neighbor[0], next_neighbor[1]] == 1:
                    self.deadlock_table[self.state_hash][location.tobytes()] = True

                    #print("case 1")
                    return True
                elif state[2, neighbor[0], neighbor[1]] == 1 and state[1, next_neighbor[0], next_neighbor[1]] == 1:
                    #print("case 2")
                    if next_neighbor in previous:
                        #depndency cycle!
                        return True
                    if self.is_frozen(state, np.array(next_neighbor), previous):
                        return True
                elif state[1, neighbor[0], neighbor[1]] == 1 and state[2, next_neighbor[0], next_neighbor[1]] == 1:
                    #print("case 3")

                    if neighbor in previous:
                        #dependency cycle!
                        return True

                    if self.is_frozen(state, np.array(neighbor), previous):
                        return True
                elif state[1, neighbor[0], neighbor[1]] == 1 and state[1, next_neighbor[0], next_neighbor[1]] == 1:
                    # print("case 4")
                    # print(neighbor in previous)
                    # print(next_neighbor in previous)
                    if neighbor in previous:
                        frozen_neighbor = True
                    else:
                        frozen_neighbor = self.is_frozen(state, np.array(neighbor), previous)
                    if next_neighbor in previous:
                        frozen_next_neighbor = True
                    else:
                        frozen_next_neighbor = self.is_frozen(state, np.array(next_neighbor), previous)


                    if frozen_neighbor and frozen_next_neighbor:
                        return True

        previous.remove(tuple(location))
        self.deadlock_table[self.state_hash][location.tobytes()] = False

        return False

    def is_dead_diagonal(self, state, location):

        directions = Environment.DIRECTIONS

        #neighbors = self.get_neighbors(location)
        #0 top right
        #1 bottom right
        #2 bottom left
        #3 top left
        diagonals = [location + directions[i] + directions[(i+1)%len(directions)] for i in range(len(directions))]
        for i in range(len(diagonals)):
            diagonal_neighbor = diagonals[i]
            diagonal_next_neighbor = diagonals[(i+1)%4]
            orientation = directions[(i+1)%len(directions)]
            across = 2*orientation + location
            center = location + orientation

            if tuple(center) in self.storage or state[0, center[0], center[1]] == 1:
                continue
            if self.is_valid(across) and ((self.state[1, center[0], center[1]] == 0).all() or (self.state[2, center[0], center[1]] == 0).all()):
                if ((self.state[1, diagonal_neighbor[0], diagonal_neighbor[1]] == 1).all() or (self.state[2, diagonal_neighbor[0], diagonal_neighbor[1]] == 1).all())\
                and ((self.state[1, diagonal_next_neighbor[0], diagonal_next_neighbor[1]] == 1).all() or (self.state[2, diagonal_next_neighbor[0], diagonal_next_neighbor[1]] == 1).all())\
                and ((self.state[1, across[0], across[1]] == 1).all() or (self.state[2, across[0], across[1]] == 1).all())  : 
                    possible_corner1 = location + directions[i]
                    possible_corner2 = location + directions[(i+2)%len(directions)]

                    opposite_corner1 = possible_corner2 + 2*orientation
                    opposite_corner2 = possible_corner1 + 2*orientation

                    corner1 = (((self.state[1, possible_corner1[0], possible_corner1[1]] == 1).all() or (self.state[2, possible_corner1[0], possible_corner1[1]] == 1).all())\
                    and ((self.state[1, opposite_corner1[0], opposite_corner1[1]] == 1).all() or (self.state[2, opposite_corner1[0], opposite_corner1[1]] == 1).all()))
                    corner2 = (((self.state[1, possible_corner2[0], possible_corner2[1]] == 1).all() or (self.state[2, possible_corner2[0], possible_corner2[1]] == 1).all())\
                    and ((self.state[1, opposite_corner2[0], opposite_corner2[1]] == 1).all() or (self.state[2, opposite_corner2[0], opposite_corner2[1]] == 1).all()))
                    if corner1 or corner2:
                        #dead diagonal! hash all valid locations and update...
                        surrounding = [diagonal_neighbor, diagonal_next_neighbor, across, possible_corner1, possible_corner2, opposite_corner1, opposite_corner2, location]

                        for place in surrounding:
                            if state[1, place[0], place[1]] == 1 and tuple(place) not in self.storage:
                                self.deadlock_table[self.state_hash][place.tobytes()] = True
                        return True
        #print(f"return false for {location}")
        return False


    def is_deadlock(self, state):
        # if not self.frozen_nodes:
        #   self.frozen_nodes = set([])
        self.state_hash = state[:2,:,:].tobytes()

        if self.state_hash not in self.deadlock_table:
            self.cache_miss += 1
            self.deadlock_table[self.state_hash] = {}
        else:
            self.cache_hit += 1

        frozen_count = 0
        for i in range(self.xlim+1):
            for j in range(self.ylim+1):
                box = np.array([i,j])
                if state[1, i, j] == 1:
                    if box.tobytes() in self.deadlock_table[self.state_hash] and self.deadlock_table[self.state_hash][box.tobytes()]:
                        frozen_count += 1
                    else:
                        if self.is_frozen(state, box, previous=set([])):
                            self.deadlock_table[self.state_hash][box.tobytes()] = True
                            frozen_count += 1
                        elif self.is_dead_diagonal(state, box):
                            frozen_count += 1

        if self.num_boxes - frozen_count < len(self.storage):
            return True

        #self.frozen_nodes = None
        return False

    # def undo(self):
    #   if self.previous_move is None:
    #       self.reset() ##no previous move? reset
    #   else:   

    #       self.state = copy.deepcopy(self.previous_move.previous_state)
    #       #print(self.state)
    #       self.has_scored = copy.deepcopy(self.previous_move.previous_scores)

    #       #undo movement
    #       self.map[tuple(self.state[0])] = PLAYER
    #       # if self.previous_move.box_moved:
    #       #   self.map[tuple(self.state[0] + self.previous_move.action)] = BOX
    #       #   self.map[tuple(self.state[0] + 2*self.previous_move.action)] = EMPTY
    #       self.reset_map()




        # for box in self.state[2:]:
        #   if not self.map[tuple(box)] == BOX:
        #       print(f"state:{self.state[0]}")
        #       print(f"previous_state:{self.previous_move.previous_state[0]}")

        #       assert self.map[tuple(box)] == BOX, f"{box} is not in MAP relative to {self.state[0]}"


    def next_state(self, state, action):
        '''
        Returns a copy with the next state.
        '''
        player = self.get_player(state)

        next_position = np.array(player) + action

        next_state = np.copy(state)

        if state[1, next_position[0], next_position[1]] == 1:
            next_box_position = next_position + action

            if state[1, next_box_position[0], next_box_position[1]] == 0 and state[2, next_box_position[0], next_box_position[1]] == 0:
                next_state[0, player[0], player[1]] = 0
                next_state[0, next_position[0], next_position[1]] = 1
                next_state[1, next_position[0], next_position[1]] = 0

                next_state[1, next_box_position[0], next_box_position[1]] = 1
                
                for index in range(len(self.boxes)):
                    if (self.boxes[index] == next_position).all():
                        #print(f"before:{self.boxes[index]}")

                        self.boxes[index] = next_box_position
                        #print(f"after:{self.boxes[index]}")

#                 for i in range(len(self.state[2:])):
#                     if (self.state[i+2] == next_position).all():
#                         self.state[i+2] = box_next_position 
#                         if tuple(box_next_position) in self.storage:
#                             self.has_scored[i] = 1.
#                         break
                

        elif state[2, next_position[0], next_position[1]] == 1:
            pass
        elif state[1, next_position[0], next_position[1]] == 0 and state[2, next_position[0], next_position[1]] == 0:
            #print("EMPTY")
            next_state[0, player[0], player[1]] = 0
            next_state[0, next_position[0], next_position[1]] = 1
            #return next_position

#         self.state[1,0] = np.sum(self.has_scored)
        return next_state

    # def step(self, evaluate=False):


    #   if not evaluate:
    #       action = self.actor.learn(State(self.state[0], self.state[1:]), self.map)
    #   else:
    #       action = self.actor.evaluate(State(self.state[0], self.state[1:]), self.map)
    #   #print(move)
    #   #print(move)
    #   next_position = action + self.state[0]
    #   #print(next_position)
        






    def draw(self, state, save_figure = False):
        #print(f"num_score:{self.state[1,0]}")
        ax = plt.gca()
        ax.clear()
        #create square boundary
        lim = max(self.xlim, self.ylim)
        plt.xlim(0, lim+1)
        plt.ylim(0, lim+1)
        ax.set_xticks(np.arange(0, lim+1))
        ax.set_yticks(np.arange(0, lim+1))
        plt.grid(alpha=0.2)


        deadlock_flag = self.is_deadlock(state)
        for i in range(self.xlim+1):
            for j in range(self.ylim+1):
                #print((i,j))
                if state[2, i,j] == 1:
                    rect = patches.Rectangle((i+0.5, j+0.5),-1,-1,linewidth=0.5,edgecolor='slategray',facecolor='slategray')
                    ax.add_patch(rect)

                elif state[0, i,j] == 1:
                    plt.plot(i, j, 'o', color='orange')
                elif state[1, i,j] == 1:  
                    nphash = np.array([i,j]).tobytes()
                    if nphash in self.deadlock_table[self.state_hash] and self.deadlock_table[self.state_hash][nphash]:
                        rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='red', facecolor='red')
                        ax.add_patch(rect)
                    else:
                        rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='tan', facecolor='tan')
                        ax.add_patch(rect)

        for place in self.storage:
            circle = patches.Circle(place, 0.05, edgecolor='limegreen', facecolor='limegreen')
            ax.add_patch(circle)



        
        #plt.draw()
        #plt.show()
        if save_figure:
            plt.savefig('sokoban.png')
        else:
            plt.show(block=False)
            # background = fig.canvas.copy_from_bbox(ax.bbox)
            # fig.canvas.restore_region(background)
            # fig.canvas.draw()

            plt.pause(self.pause)
