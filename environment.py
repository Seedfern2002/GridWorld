import numpy as np
from copy import deepcopy

WORLD_SIZE = 10
cities = [[0, 4], [2, 1], [2, 3], [3, 5], [6, 2], [7, 7], [9, 5]]
CITY_REWARD = -10


class Env:
    def __init__(self):
        # init the grid world
        self.height = WORLD_SIZE
        self.width = WORLD_SIZE
        self.world = np.zeros((self.height, self.width))

        # the coordinate of cities
        self.cities = deepcopy(cities)
        self.num_cities = len(self.cities)
        self.city_flag = (1 << self.num_cities) - 1

        # set the Rewards in different location
        self.world = self.world - 10

        for city in self.cities:
            self.world[city[0], city[1]] = CITY_REWARD

        # self.world[2, 1] = -10
        # self.world[6, 2] = -10

        # set the initial state and the available actions
        self.current_state = [0, 0, 0]
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.terminal = False

        self.out_reward = -500
        self.update = False

    def reset(self):
        self.__init__()

    def get_reward(self, state):
        return self.world[state[0], state[1]]

    def step(self, action):
        # update state according to the action agent chooses
        last_state = self.current_state

        if action == 'UP':
            if last_state[0] == 0:
                reward = self.out_reward
                self.terminal = True
            else:
                self.current_state[0] -= 1
                reward = self.get_reward(self.current_state)

        elif action == 'DOWN':
            if last_state[0] == self.height - 1:
                reward = self.out_reward
                self.terminal = True
            else:
                self.current_state[0] += 1
                reward = self.get_reward(self.current_state)

        elif action == 'LEFT':
            if last_state[1] == 0:
                reward = self.out_reward
                self.terminal = True
            else:
                self.current_state[1] -= 1
                reward = self.get_reward(self.current_state)

        elif action == 'RIGHT':
            if last_state[1] == self.width - 1:
                reward = self.out_reward
                self.terminal = True
            else:
                self.current_state[1] += 1
                reward = self.get_reward(self.current_state)

        # check if an unarrived city is reached
        for i, city in enumerate(self.cities):
            if city[0] == self.current_state[0] and city[1] == self.current_state[1]:
                city_flag = 1 << i
                if self.current_state[2] & city_flag != 0:
                    self.terminal = True
                else:
                    self.current_state[2] |= city_flag
                    self.world[city[0], city[1]] = self.out_reward
                break
        # self.world[city[0], city[1]] = self.out_reward
        if self.current_state[2] == self.city_flag and not self.update:
            self.world[0, 0] = 10000
        return reward

    def check_terminal(self):
        if self.terminal:
            return 'TERMINAL'

        # if all the cities have been reached and agent's coordinate is (0, 0)
        if self.current_state[0] == 0 and self.current_state[1] == 0 and self.current_state[2] == self.city_flag:
            return 'TERMINAL'
        else:
            return 'RUNNING'
