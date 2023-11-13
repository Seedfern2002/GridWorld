import os
import argparse
from Q_learning import Agent as Q_Agent
from policy_gradient import Agent as PG_Agent
from environment import Env

parser = argparse.ArgumentParser(description='Agent in Grid World')
parser.add_argument('--Agent_type', default='Q',
                    type=str, help='The agent type to evaluate')
parser.add_argument('--Q_path', default='./parameter3/Q_table.npy',
                    type=str, help='The Q-table path')
parser.add_argument('--PG_path', default='./parameter3/PG_theta.pth',
                    type=str, help='The PG parameter theta path')
parser.add_argument('--output_dir', default='',
                    type=str, help='Where to store the state-action sequence of agent, default is printing it')
args = parser.parse_args()


def eval_agents(env, agent):
    env.reset()
    episode = [[0, 0]]
    actions = []
    step = 0
    terminal = False

    i = 0
    while not terminal and i < 100:
        i += 1
        action = agent.act()
        _ = env.step(action)
        current_state = env.current_state
        coordinate = [current_state[0], current_state[1]]

        actions.append(action)
        episode.append(coordinate)
        step += 1
        if env.check_terminal() == 'TERMINAL':
            terminal = True
    if i == 100:
        print('Failed to reach goals')

    return episode, actions


def main():
    environment = Env()
    # loading agent
    if args.Agent_type == 'Q':
        agent = Q_Agent(environment, epsilon=0.)
        if os.path.exists(args.Q_path):
            agent.load(args.Q_path)
        else:
            print('\'Q_path\' not exists!')
    elif args.Agent_type == 'PG':
        agent = PG_Agent(environment, learning=False)
        if os.path.exists(args.PG_path):
            agent.load(args.PG_path)
        else:
            print('\'PG_path\' not exists!')
    else:
        print('\'Agent_type\' parameter needs \'Q\' or \'PG\' as input')
        return
    episode, actions = eval_agents(environment, agent)
    steps = len(actions)
    if args.output_dir == '':
        print('The Agent made %d moves' % steps)
        for coordinate, action in zip(episode, actions):
            print(f'At ({coordinate[0]}, {coordinate[1]}), the agent chose {action}')
    else:
        dire = args.output_dir
        if not os.path.exists(dire):
            os.makedirs(dire)
        f = open(dire + '/' + args.Agent_type + '_state_action_sequence' + '.txt', 'w')
        f.write('The Agent made %d moves\n' % steps)
        for coordinate, action in zip(episode, actions):
            f.write(f'At ({coordinate[0]}, {coordinate[1]}), the agent chose {action}\n')


if __name__ == '__main__':
    main()
