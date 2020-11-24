class Environment:
    """
    This class keeps all map information, such as width, positions of goals, from input file.
    """

    def __init__(self, input_text):
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


class State:
    """
    This class maintain the state that may be updated during the game.
    """

    def __init__(self, boxes, player_position):
        self.boxes = boxes
        self.player_position = player_position
