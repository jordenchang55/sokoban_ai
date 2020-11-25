from __future__ import annotations


class Action:
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    def __init__(self, direction, steps: int):
        self.steps = steps
        self.direction = direction


class State:
    """
    This class maintain the state that may be updated during the game.
    """

    def __init__(self, boxes: [int], player_position: int):
        self.boxes = boxes
        self.player_position = player_position

    def __eq__(self, other):
        return self.player_position == other.player_position and set(self.boxes) == set(other.boxes)


class Environment:
    """
    This class keeps all map information, such as width, positions of goals, from input file.
    """

    def __init__(self, input_text: str):
        self.storages = []
        self.walls = []
        self.width = 0
        self.height = 0
        # Deadlocks are not for drawing on map but for improving the performance.
        # TODO: pre-compute the deadlock spots.
        self.deadlocks = []

        boxes = []
        player = 0
        for index, row in enumerate(input_text):

            if index == 0:
                self.width = int(row[0])
                self.height = int(row[1])

            if index == 1:
                self.walls = self._flatten(row)
            elif index == 2:
                boxes = self._flatten(row)
            elif index == 3:
                self.storages = self._flatten(row)
            elif index == 4:
                player = self._flatten([0] + row)[0]

        self.state = State(boxes, player)

    def _flatten(self, points):
        return [self.width * (int(points[i]) - 1) + int(points[i + 1]) for i in range(1, len(points), 2)]

    def _unflatten(self, positions):
        return [((pos - 1) % self.width + 1, int((pos - 1) / self.width) + 1) for pos in positions]

    def draw(self, save_figure=False):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        ax = plt.gca()

        # create square boundary
        lim = max(self.width, self.height)
        plt.xlim(-1, lim + 1)
        plt.ylim(-1, lim + 1)
        ax.set_xticks(np.arange(-1, lim + 1))
        ax.set_yticks(np.arange(lim + 1, 0))
        plt.grid(alpha=0.2)

        for wall in self._unflatten(self.walls):
            rect = patches.Rectangle(wall, -1, -1, linewidth=1, edgecolor='slategray',
                                     facecolor='slategray')
            ax.add_patch(rect)

        for box in self._unflatten(self.state.boxes):
            rect = patches.Rectangle(box, -1, -1, linewidth=1, edgecolor='tan', facecolor='tan')
            ax.add_patch(rect)

        for place in self._unflatten(self.storages):
            circle = patches.Circle((place[0] - 0.5, place[1] - 0.5), 0.1, edgecolor='limegreen', facecolor='limegreen')
            ax.add_patch(circle)
        player_point = self._unflatten([self.state.player_position])[0]
        plt.plot(player_point[0] - 0.5, player_point[1] - 0.5, 'o', color='orange')

        plt.draw()
        if save_figure:
            plt.savefig('sokoban.png')
        else:
            plt.show()

    def is_goal(self, node: Node) -> bool:
        """
        Determine if all boxes are put onto storages.
        """
        return set(self.storages) == set(node.state.boxes)

    def get_actions(self, state: State) -> [Action]:
        """
        Gets available actions from given state.
        """
        # TODO: implement this
        pass

    def move(self, state, action) -> State:
        # TODO: implement this
        pass

    def calc_action_cost(self, state: State, action: Action, next_state: State) -> int:
        # TODO: implement this
        pass


class Node:
    """
    This class represents the node in search tree and it contains the state and
    """

    def __init__(self, state: State, action: Action = None, parent: Node = None, path_cost: int = 0):
        self.state = state
        self.action = action
        self.parent = parent
        self.path_cost = path_cost

    def __cmp__(self, other):
        """
        This is comparing function for priority queue to decide which node will go first.
        :param other: another node compares to this one.
        """
        return Node.get_value(self) - Node.get_value(other)

    @staticmethod
    def get_value(node: Node):
        # This needs to be override
        raise NotImplementedError
