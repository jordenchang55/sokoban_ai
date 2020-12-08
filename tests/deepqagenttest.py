



import unittest
import numpy as np
import csv
from environment import Environment
from environment import DOWN
from environment import RIGHT
from environment import LEFT
from environment import UP

from deepqagent import DeepQAgent
import copy
#import logging





class DeepQAgentTest(unittest.TestCase):




    def setUp(self):
        with open('inputs/sokoban00.txt', 'r') as file:
            csv_input = csv.reader(file, delimiter=' ')

            for index, row in enumerate(csv_input):

                def unpack(points):

                    return [tuple([int(points[index+1]), int(points[index])]) for index in range(1, len(points), 2)]

                #print(index, row)

                if index == 0:
                    #sizeH, sizeV

                    xlim = int(row[0])
                    ylim = int(row[1])
                if index == 1:
                    #print(MapType.WALL.value)
                    walls = unpack(row)
                elif index == 2:    
                    boxes = unpack(row)
                elif index == 3:
                    storage = unpack(row)
                elif index == 4:
                    player = np.array([int(row[0]), int(row[1])])

        environment = Environment(walls = walls, boxes = boxes, storage = storage, player = player, xlim = xlim, ylim = ylim)
        self.agent = DeepQAgent(environment = environment, learning_rate = 1.0, discount_factor=0.95, verbose=False)


    def tearDown(self):
        self.agent = None



    def test_reward(self):
        pass


