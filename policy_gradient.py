import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from environment import Env
from torch.distributions import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

save_dir = './parameter3'
name = 'PG_theta.pth'
fig_dir = './learning_fig3'
torch.manual_seed(666)


class Agent:
    def __init__(self, env, lr=0.02, batch_size=100, gamma=0.98, learning=True):
        self.env = env
        self.theta = torch.zeros((self.env.height, self.env.width, self.env.city_flag + 1, 4),
                                 requires_grad=True)
        self.lr = lr
        self.batch_size = batch_size
        self.loss = 0.
        self.counter = 0
        self.gamma = gamma
        self.learning = learning
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_number = {'UP': 0,
                              'DOWN': 1,
                              'LEFT': 2,
                              'RIGHT': 3}

        # store the history
        self.log_probs = []
        self.rewards = []
        self.running_average = 0.
        self.rand_prob = torch.tensor([0.05, 0.05, 0.05, 0.05])

        self.optimizer = optim.SGD([self.theta], lr=self.lr)

    def policy(self, state):
        return F.softmax(self.theta[state[0], state[1], state[2]], dim=0)   # + self.rand_prob

    def act(self):
        state = self.env.current_state
        probs = self.policy(state)
        if self.learning:
            # sampling from the probability distribution probs
            m = Categorical(probs)
            action_number = m.sample()
            self.log_probs.append(m.log_prob(action_number))
            action = self.actions[action_number.item()]
        else:
            # if not learning, select the action which has the largest prob
            action_number = torch.argmax(probs, dim=0)
            action = self.actions[action_number.item()]
        return action

    def learn(self):
        accumulated_rewards = []
        ar = 0.
        # calculate the accumulated rewards of each state rather than using the total rewards
        for r in self.rewards[::-1]:
            ar = r + self.gamma * ar
            accumulated_rewards.insert(0, ar)
        # calculate the running average(biased estimate)
        self.running_average = (1 - self.lr) * self.running_average + \
                               self.lr * (np.mean(accumulated_rewards) - self.running_average)
        # using normalized rewards
        accumulated_rewards -= self.running_average
        accumulated_rewards = torch.tensor(accumulated_rewards)
        for log_prob, ar in zip(self.log_probs, accumulated_rewards):
            # maximize log_prob*ar equal to minimize -log_prob*ar
            self.loss += -log_prob * ar
        del self.rewards[:]
        del self.log_probs[:]

        self.counter += 1
        if self.counter == self.batch_size:
            # update parameters per batch_size episodes
            self.counter = 0
            self.backward()
        return

    def backward(self):
        self.optimizer.zero_grad()
        self.loss = self.loss / self.batch_size
        self.loss.backward()
        self.optimizer.step()
        self.loss = 0.

    def save(self):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.theta, save_dir + '/' + name)

    def load(self, path):
        print('loading parameters from %s' % path)
        self.theta = torch.load(path)
        print('parameters has been loaded')


def play(env, agent, num_episodes=5000, max_steps_per_episode=1000, learning=True):
    reward_per50_episode = []
    average_reward_per50_episode = []
    rewards_50 = []
    running_average = []
    average = 0.
    for episode in range(1, num_episodes + 1):
        if episode % 500 == 0:
            agent.save()
        env.reset()
        rewards = 0.
        step = 0
        terminal = False
        while step < max_steps_per_episode and not terminal:
            action = agent.act()
            reward = env.step(action)
            agent.rewards.append(reward)  # save the reward information

            rewards += reward
            step += 1

            if env.check_terminal() == 'TERMINAL':
                terminal = True

        if learning:
            agent.learn()

        average = average + (rewards - average) / episode
        running_average.append(average)
        rewards_50.append(rewards)

        if episode % 50 == 49:
            reward_per50_episode.append(rewards)
            average_reward_per50_episode.append(np.mean(rewards_50))
            rewards_50.clear()
            print(f'epoch: %07d steps: %d, rewards: %.04f, city: %s' % (episode, step,
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
    plt.savefig(fig_dir + '/PG_rewards.png')
    plt.clf()

    plt.plot(running_average_rewards)
    plt.xlabel("num_episode")
    plt.ylabel("running average")
    plt.title("running average rewards curve")
    plt.savefig(fig_dir + '/PG_running_average_rewards.png')
    plt.clf()

    plt.plot(average_rewards)
    plt.xlabel("num_episode/50")
    plt.ylabel("average rewards in 50 episodes")
    plt.title("average rewards curve")
    plt.savefig(fig_dir + '/PG_average_rewards.png')
    plt.clf()
