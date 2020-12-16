import copy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from environment import Environment


class State:
    EMPTY = 0
    PLAYER = 1
    BOX = 2
    WALL = 3

    def __init__(self, walls, boxes, player, storage, xlim, ylim):
        self.map = np.zeros((xlim + 1, ylim + 1))

        for wall in walls:
            self.map[wall] = self.WALL
        for box in boxes:
            self.map[box] = self.BOX

        self.map[tuple(player)] = self.PLAYER
        self.player = player

        self.storage = set(storage)
        self.boxes = np.array(boxes)

        # self.max_score = 0

    def __hash__(self):
        return self.map.tobytes()

    def copy(self):
        """
        Return a value-copied instance.
        """
        s = State(walls=[], boxes=[], player=(0, 0), storage=self.storage, xlim=0, ylim=0)
        s.map = np.copy(self.map)
        s.player = np.copy(self.player)
        s.boxes = np.copy(self.boxes)
        s.storage = self.storage.copy()
        # s.max_score = self.max_score
        return s


class StateEnvironment(Environment):

    def __init__(self, filename, walls, boxes, player, storage, xlim, ylim, pause=0.05):
        super().__init__(filename, xlim, ylim)

        self.state = State(walls, boxes, player, storage, xlim, ylim)

        self.pause = pause

        self.deadlock_table = {}

        self.original_state = copy.deepcopy(self.state)

        self.state_hash = None

        self.cache_miss = 0
        self.cache_hit = 0

    def reset(self):
        # print("reset!")
        # print(f"player:{self.state[0]}")
        # print(f"reset_player:{self.original_player}")
        self.state = copy.deepcopy(self.original_state)

    def is_valid(self, location):
        x, y = location

        return (x >= 0 and x <= self.xlim and y >= 0 and y <= self.ylim)

    def count_boxes_scored(self, state):

        return sum([1 if state.map[place] == State.BOX else 0 for place in state.storage])
        # count = 0
        # for place in state.storage:
        #     if state.map[place] == State.BOX:
        #         count += 1
        # return count

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
        ''''
        Detects if a given box in a state in frozen deadlock.


        previous: the set of edges for dependencies. (a depends b)

        '''
        if location.tobytes() in self.deadlock_table[self.state_hash]:
            return self.deadlock_table[self.state_hash][location.tobytes()]

        location_tuple = tuple(location)
        neighbors = self.get_neighbors(location)

        state_map = state.map
        for i in range(len(neighbors)):
            neighbor = tuple(neighbors[i])
            next_neighbor = tuple(neighbors[(i + 1) % len(neighbors)])

            # if location_tuple == (9,3) or location_tuple == (9, 4) or location_tuple == (8, 4):
            #     print(f"neighbor:{neighbor}")
            #     print(f"next_neighbor:{next_neighbor}")

            #     print(f"frozen_neighbor:{(neighbor, location_tuple) in previous}")
            #     print(f"frozen_next_neighbor:{(next_neighbor, location_tuple) in previous}")
            if state_map[neighbor] == State.WALL and state_map[next_neighbor] == State.WALL:
                self.deadlock_table[self.state_hash][location.tobytes()] = True
                return True
            elif state_map[neighbor] == State.WALL and state_map[next_neighbor] == State.BOX:
                previous.add((location_tuple, next_neighbor))
                if (next_neighbor, location_tuple) in previous:
                    # depndency cycle!
                    return True
                if self.is_frozen(state, np.array(next_neighbor), previous):
                    return True
                previous.remove((location_tuple, next_neighbor))

            elif state_map[neighbor] == State.BOX and state_map[next_neighbor] == State.WALL:
                previous.add((location_tuple, neighbor))
                # print("case 3")

                if (neighbor, location_tuple) in previous:
                    # dependency cycle!
                    return True

                if self.is_frozen(state, np.array(neighbor), previous):
                    return True
                previous.remove((location_tuple, neighbor))

            elif state_map[neighbor] == State.BOX and state_map[next_neighbor] == State.BOX:

                previous.add((location_tuple, neighbor))
                previous.add((location_tuple, next_neighbor))

                frozen_neighbor = True if (neighbor, location_tuple) in previous else self.is_frozen(state,
                                                                                                     np.array(neighbor),
                                                                                                     previous)
                frozen_next_neighbor = True if (next_neighbor, location_tuple) in previous else self.is_frozen(state,
                                                                                                               np.array(
                                                                                                                   next_neighbor),
                                                                                                               previous)

                # ax = plt.gca()

                # rect = patches.Rectangle((neighbor[0]+0.5, neighbor[1]+0.5), -1, -1, linewidth=0.5, edgecolor='deepskyblue')
                # ax.add_patch(rect)
                # rect = patches.Rectangle((next_neighbor[0]+0.5, next_neighbor[1]+0.5), -1, -1, linewidth=0.5, edgecolor='deepskyblue')
                # ax.add_patch(rect)

                # plt.show(block=True)

                if frozen_neighbor and frozen_next_neighbor:
                    return True

                previous.remove((location_tuple, neighbor))
                previous.remove((location_tuple, next_neighbor))

        return False

    def is_dead_diagonal(self, state, location):
        state_map = state.map

        directions = Environment.DIRECTIONS

        # neighbors = self.get_neighbors(location)
        # 0 top right
        # 1 bottom right
        # 2 bottom left
        # 3 top 
        if tuple(location) in state.storage:
            return False
        diagonals = [location + directions[i] + directions[(i + 1) % len(directions)] for i in range(len(directions))]
        for i in range(len(diagonals)):
            diagonal_neighbor = tuple(diagonals[i])
            diagonal_next_neighbor = tuple(diagonals[(i + 1) % 4])
            orientation = directions[(i + 1) % len(directions)]
            across = tuple(2 * orientation + location)
            center = location + orientation

            if tuple(center) in state.storage:
                continue

            if self.is_valid(across) and state_map[tuple(center)] == State.EMPTY:
                if state_map[diagonal_neighbor] > State.PLAYER and state_map[diagonal_next_neighbor] > State.PLAYER and state_map[across] > State.PLAYER:  # hack for checking if its box or wall
                    possible_corner1 = location + directions[i]
                    possible_corner2 = location + directions[(i + 2) % len(directions)]

                    opposite_corner1 = possible_corner2 + 2 * orientation
                    opposite_corner2 = possible_corner1 + 2 * orientation

                    if ((state_map[tuple(possible_corner1)] > State.PLAYER and state_map[
                        tuple(opposite_corner1)] > State.PLAYER) or (
                            state_map[tuple(possible_corner2)] > State.PLAYER and state_map[
                        tuple(opposite_corner2)] > State.PLAYER)):
                        # dead diagonal! hash all valid locations and update...
                        surrounding = [diagonal_neighbor, diagonal_next_neighbor, across, possible_corner1,
                                       possible_corner2, opposite_corner1, opposite_corner2, location]

                        for place in np.array(surrounding):
                            if state_map[tuple(place)] == State.BOX and tuple(place) not in state.storage:
                                self.deadlock_table[self.state_hash][place.tobytes()] = True

                        return True
        # print(f"return false for {location}")
        return False

    def is_deadlock(self, state):

        # if self.cache_miss != 0 and self.cache_hit != 0:
        # print(f"deadlock_table_rate:{self.cache_hit/(self.cache_hit + self.cache_miss)}")

        # if not self.frozen_nodes:
        #   self.frozen_nodes = set([])

        self.state_hash = state.map.tobytes()

        if self.state_hash not in self.deadlock_table:
            self.cache_miss += 1
            self.deadlock_table[self.state_hash] = {}
        else:
            self.cache_hit += 1

        frozen_count = 0
        for box in state.boxes:
            #print(box)
            box_hash = box.tobytes()
            if tuple(box) not in state.storage:
                if box_hash in self.deadlock_table[self.state_hash]:
                    if self.deadlock_table[self.state_hash][box_hash]:
                        frozen_count += 1
                else:
                    if self.is_frozen(state, box, previous=set([])):
                        self.deadlock_table[self.state_hash][box.tobytes()] = True
                        frozen_count += 1
                    elif self.is_dead_diagonal(state, box):
                        frozen_count += 1

        return (len(state.boxes) - frozen_count < len(state.storage))

    # if len(state.boxes) - frozen_count < len(state.storage):
    #     return True
    # return False

    def next_state(self, state, action):
        '''
        Returns a copy with the next state.
        '''
        player = self.get_player(state)

        next_position = np.array(player) + action

        next_state = state.copy()

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
            # print("EMPTY")
            next_state.map[tuple(player)] = State.EMPTY
            next_state.map[tuple(next_position)] = State.PLAYER
            next_state.player = next_position
        # return next_position

        #score_count = self.count_boxes_scored(next_state)
        # if next_state.max_score < score_count:
        #     next_state.max_score = score_count

        return next_state

    def draw(self, state, save_figure=False):
        # print(f"num_score:{self.state[1,0]}")
        ax = plt.gca()
        ax.clear()
        # create square boundary
        lim = max(self.xlim, self.ylim)
        plt.xlim(0, lim + 1)
        plt.ylim(0, lim + 1)
        ax.set_xticks(np.arange(0, lim + 1))
        ax.set_yticks(np.arange(0, lim + 1))
        plt.grid(alpha=0.2)

        deadlock_flag = self.is_deadlock(state)
        for i in range(self.xlim + 1):
            for j in range(self.ylim + 1):
                # print((i,j))
                if state.map[i, j] == State.WALL:
                    rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='slategray',
                                             facecolor='slategray')
                    ax.add_patch(rect)

                elif state.map[i, j] == State.PLAYER:
                    plt.plot(i, j, 'o', color='orange')
                elif state.map[i, j] == State.BOX:
                    nphash = np.array([i, j]).tobytes()
                    if deadlock_flag and nphash in self.deadlock_table[self.state_hash] and \
                            self.deadlock_table[self.state_hash][nphash]:
                        rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='red',
                                                 facecolor='red')
                        ax.add_patch(rect)
                    else:
                        rect = patches.Rectangle((i + 0.5, j + 0.5), -1, -1, linewidth=0.5, edgecolor='tan',
                                                 facecolor='tan')
                        ax.add_patch(rect)

        for place in state.storage:
            circle = patches.Circle(place, 0.05, edgecolor='limegreen', facecolor='limegreen')
            ax.add_patch(circle)

        # plt.draw()
        # plt.show()
        if save_figure:
            plt.savefig('sokoban.png')
        else:
            plt.show(block=False)
            # background = fig.canvas.copy_from_bbox(ax.bbox)
            # fig.canvas.restore_region(background)
            # fig.canvas.draw()

            plt.pause(self.pause)
