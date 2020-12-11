import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from environment import Environment


class State():
    EMPTY = 0
    PLAYER = 1
    BOX = 2
    WALL = 3

    def __init__(self, walls, boxes, player, storage, xlim, ylim):
        self.map = np.zeros((xlim+1, ylim+1))

        for wall in walls:
            self.map[wall] = self.WALL
        for box in boxes:
            self.map[box] = self.BOX

        self.map[tuple(player)] = self.PLAYER
        self.player = player

        self.storage = set(storage)
        self.boxes = np.array(boxes)

    def __hash__(self):
        return self.state.map.tobytes()



class StateEnvironment(Environment):


    def __init__(self, walls, boxes, player, storage, xlim, ylim, pause=0.05):
        super().__init__(xlim, ylim)

        self.state = State(walls, boxes, player, storage, xlim, ylim)

        self.pause = pause

        self.deadlock_table = {}

        self.original_state = copy.deepcopy(self.state)
        
        
        self.state_hash = None

        self.cache_miss = 0
        self.cache_hit = 0

        #self.draw(self.state)


    def reset(self):
        # print("reset!")
        # print(f"player:{self.state[0]}")
        # print(f"reset_player:{self.original_player}")
        self.state = copy.deepcopy(self.original_state)


    def is_valid(self, location):
        x, y = location

        return (x >= 0 and x <= self.xlim and y >= 0 and y <= self.ylim)

    def count_boxes_scored(self, state):
        count = 0
        for place in state.storage:
            if state.map[place] == State.BOX:
                count += 1
        return count

    def is_goal_state(self, state):
        for place in state.storage:
            if state.map[place] != State.BOX:
                return False

        return True

    def get_player(self, state):
        return state.player
    def get_neighbors(self, location):
        return [location + direction for direction in Environment.DIRECTIONS]


    def is_frozen(self, state, location, previous=None):

        if location.tobytes() in self.deadlock_table[self.state_hash]:
          return self.deadlock_table[self.state_hash][location.tobytes()]


        neighbors = self.get_neighbors(location)
        previous.add(tuple(location))
        if tuple(location) not in state.storage:
            for i in range(len(neighbors)):
                neighbor = tuple(neighbors[i])
                next_neighbor = tuple(neighbors[(i+1)%len(neighbors)])


                if state.map[neighbor] == State.WALL and state.map[next_neighbor] == State.WALL:
                    self.deadlock_table[self.state_hash][location.tobytes()] = True
                    return True
                elif state.map[neighbor] == State.WALL and state.map[next_neighbor] == State.BOX:
                    #print("case 2")
                    if next_neighbor in previous:
                        #depndency cycle!
                        self.deadlock_table[self.state_hash][location.tobytes()] = True
                        return True
                    if self.is_frozen(state, np.array(next_neighbor), previous):
                        self.deadlock_table[self.state_hash][location.tobytes()] = True
                        return True
                elif state.map[neighbor] == State.BOX and state.map[next_neighbor] == State.WALL:
                    #print("case 3")

                    if neighbor in previous:
                        #dependency cycle!
                        self.deadlock_table[self.state_hash][location.tobytes()] = True
                        return True

                    if self.is_frozen(state, np.array(neighbor), previous):
                        self.deadlock_table[self.state_hash][location.tobytes()] = True
                        return True
                elif state.map[neighbor] == State.BOX and state.map[next_neighbor] == State.BOX:
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
                        self.deadlock_table[self.state_hash][location.tobytes()] = True
                        return True

        previous.remove(tuple(location))
        self.deadlock_table[self.state_hash][location.tobytes()] = False

        return False


    def is_deadlock(self, state):
        
        #if self.cache_miss != 0 and self.cache_hit != 0:
            #print(f"deadlock_table_rate:{self.cache_hit/(self.cache_hit + self.cache_miss)}")

        # if not self.frozen_nodes:
        #   self.frozen_nodes = set([])
        self.state_hash = state.boxes.tobytes()

        if self.state_hash not in self.deadlock_table:
            self.cache_miss += 1
            self.deadlock_table[self.state_hash] = {}
        else:
            self.cache_hit += 1
        for box in state.boxes:
            if box.tobytes() in self.deadlock_table[self.state_hash] and self.deadlock_table[self.state_hash][box.tobytes()]:
                return True
            elif self.is_frozen(state, box, previous=set([])):
                return True


        #self.frozen_nodes = None
        return False


    def next_state(self, state, action):
        '''
        Returns a copy with the next state.
        '''
        player = self.get_player(state)

        next_position = np.array(player) + action

        next_state = copy.deepcopy(state)

        if state.map[tuple(next_position)] == State.BOX:
            next_box_position = next_position + action

            if state.map[tuple(next_box_position)] == State.EMPTY:
                next_state.map[tuple(player)] = State.EMPTY
                next_state.player = next_position
                next_state.map[tuple(next_position)] = State.PLAYER
                next_state.map[tuple(next_box_position)] = State.BOX

                for index in range(len(next_state.boxes)):
                    if (next_state.boxes[index] == next_position).all():

                        next_state.boxes[index] = next_box_position

                

        elif state.map[tuple(next_position)] == State.WALL:
            pass
        elif state.map[tuple(next_position)] == State.EMPTY:
            #print("EMPTY")
            next_state.map[tuple(player)] = State.EMPTY
            next_state.map[tuple(next_position)] = State.PLAYER
            next_state.player = next_position
            #return next_position

#         self.state[1,0] = np.sum(self.has_scored)
        # for box in state.boxes:
        #     assert state.map[tuple(box)] == State.BOX, "boxes misaligned with map."

        return next_state

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
                if state.map[i, j] == State.WALL:
                    rect = patches.Rectangle((i+0.5, j+0.5),-1,-1,linewidth=0.5,edgecolor='slategray',facecolor='slategray')
                    ax.add_patch(rect)

                elif state.map[i,j] == State.PLAYER:
                    plt.plot(i, j, 'o', color='orange')
                elif state.map[i, j] == State.BOX:  
                    nphash = np.array([i,j]).tobytes()
                    if deadlock_flag and nphash in self.deadlock_table[self.state_hash] and self.deadlock_table[self.state_hash][nphash]:
                        rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='red', facecolor='red')
                        ax.add_patch(rect)
                    else:
                        rect = patches.Rectangle((i+0.5, j+0.5), -1, -1, linewidth=0.5, edgecolor='tan', facecolor='tan')
                        ax.add_patch(rect)

        for place in state.storage:
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
