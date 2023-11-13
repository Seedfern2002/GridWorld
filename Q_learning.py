import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from environment import Env

save_dir = './parameter3'
name = 'Q_table.npy'
fig_dir = './learning_fig3'

np.random.seed(666)


class Agent:
    def __init__(self, env, epsilon=0.1, alpha=0.5, gamma=0.5):
        self.env = env
        self.table = np.zeros((self.env.height, self.env.width, self.env.city_flag + 1, 4), dtype=np.float64)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_number = {'UP': 0,
                              'DOWN': 1,
                              'LEFT': 2,
                              'RIGHT': 3}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def act(self):
        if np.random.uniform(0, 1) < self.epsilon:
            # explore
            action = self.env.actions[np.random.randint(0, 4)]
        else:
            # select action according to q_value
            q_values = self.table[self.env.current_state[0]][self.env.current_state[1]][self.env.current_state[2]]
            max_value = np.max(q_values)
            action = self.actions[np.random.choice([i for i, v in enumerate(q_values) if v == max_value])]
        return action

    def learn(self, last_state, current_state, reward, action):
        # update q-table
        # bug?
        q_value_current_state = self.table[current_state[0]][current_state[1]][current_state[2]]
        q_value_last_state_action = self.table[last_state[0]][last_state[1]][last_state[2]][self.action_number[action]]
        self.table[last_state[0]][last_state[1]][last_state[2]][self.action_number[action]] = \
            (1 - self.alpha) * q_value_last_state_action + self.alpha * \
            (reward + self.gamma * np.max(q_value_current_state))

    def scheduler(self):
        # to measure that \sum alpha^2 < infinite
        self.alpha *= 0.99

    def save(self):
        # save parameters
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + '/' + name, self.table)

    def load(self, path):
        # load parameters from path
        print('loading parameters from %s' % path)
        self.table = np.load(path)
        print('parameters has been loaded')


def play(env, agent, num_episodes=30000, max_steps_per_episode=5000, learning=True):
    reward_per50_episode = []
    average_reward_per50_episode = []
    rewards_50 = []
    running_average = []
    average = 0.
    for episode in range(1, num_episodes + 1):
        if episode % 100 == 1:
            agent.save()
        if episode % 1000 == 1:
            agent.scheduler()
        env.reset()
        rewards = 0.
        step = 0
        terminal = False
        while step < max_steps_per_episode and not terminal:
            # agent interacts with the environment
            last_state = deepcopy(env.current_state)
            action = agent.act()
            reward = env.step(action)
            current_state = deepcopy(env.current_state)

            if learning:
                agent.learn(last_state, current_state, reward, action)

            rewards += reward
            step += 1

            if env.check_terminal() == 'TERMINAL':
                terminal = True

        average = average + (rewards - average)/episode
        running_average.append(average)
        rewards_50.append(rewards)
        if episode % 50 == 1:
            reward_per50_episode.append(rewards)
            average_reward_per50_episode.append(np.mean(rewards_50))
            rewards_50.clear()
            print(f'episode: %05d steps: %d, rewards: %.04f, city: %s' % (episode, step,
                                                                          rewards, bin(env.current_state[2])))

    return reward_per50_episode, running_average, average_reward_per50_episode


if __name__ == '__main__':
    environment = Env()
    agent = Agent(environment)

    rewards_list, running_average_rewards, average_rewards = play(environment, agent)

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    plt.plot(rewards_list)
    plt.xlabel("num_episode/50")
    plt.ylabel("rewards")
    plt.title("rewards curve")
    plt.savefig(fig_dir + '/Q_rewards.png')
    plt.clf()

    plt.plot(running_average_rewards)
    plt.xlabel("num_episode")
    plt.ylabel("running average")
    plt.title("running average rewards curve")
    plt.savefig(fig_dir + '/Q_running_average_rewards.png')
    plt.clf()

    plt.plot(average_rewards)
    plt.xlabel("num_episode/50")
    plt.ylabel("average rewards in 50 episodes")
    plt.title("average rewards curve")
    plt.savefig(fig_dir + '/Q_average_rewards.png')
    plt.clf()
