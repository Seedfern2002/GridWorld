## Welcome to Grid World!
The goal of the agent is to reach all cities ([0, 4], [2, 1], [2, 3], [3, 5], [6, 2], [7, 7], [9, 5]) in grid world (a 10*10 matrix) only once and get back to the starting place([0, 0]) using minimum moves.  
In this project, the agents are implemented using Q-learning and Policy Gradient respectively.  
Here is a brief introduction about how to use it:  
 
#### 1. File description
- [environment.py](environment.py) builds the environment of the grid world 
- [Q_learning.py](Q_learning.py) is the Q-learning agent code
- [policy_gradient.py](policy_gradient.py) is the Policy Gradient agent code
- [eval.py](eval.py) is the evaluate tools to output the state-action sequence of agent
- [GridWorld.pdf](GridWorld.pdf) is the detailed description of the project
- 'path' is the directory storing the state-action sequences of agents
- 'parameter3' stores the trained Q-table of Q agent and the parameters of Policy Gradient agent(local optimal)
- 'learning_fig3' stores the reward curves in training
#### The following two directories store models with prior knowledge as described in GridWorld.pdf, which can achieve global optimality 
- 'parameter' stores the trained Q-table of Q agent and the parameters of Policy Gradient agent
- 'parameter2' stores another possible parameters trained in one terminal state environment
- 'learning_fig' stores the reward curves in training
- 'learning_fig2' stores the reward curves which agents were trained in one terminal state environment

#### 2. Install the dependency
```bash
pip install -r requirements.txt
```

#### 3. Train：
##### Note: Please change variable CITY_REWARD in [environment.py](environment.py) to -10 before training Q learning Agent, 666 before training Policy Gradient Agent.
To train the Q-learning agent, using
```bash
python Q_learning.py 
```
the Q-table will be saved at ./parameter3/Q_table.npy

To train the Policy-Gradient agent, using
```bash
python policy_gradient.py
```
the parameters of agent will be saved at ./parameter3/PG_theta.pth

#### 4. Evaluate
To evaluate the Q-learning agent, using
```bash
python eval.py --Agent_type Q
```
To evaluate the Policy-Gradient agent, using
```bash
python eval.py --Agent_type PG
```
To see detailed command line arguments, using
```bash
python eval.py -h
```