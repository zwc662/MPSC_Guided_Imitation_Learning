import matplotlib.pyplot as plt
from mpc import MPC
import numpy as np
import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]

policy_net, _, running_state = pickle.load(open("../assets/learned_models/CartPole-v0_ppo.p", "rb"))
#env.env.theta_threshold_radians = np.pi * 2
#env.env.x_threshold = 5 

start_theta = 0#-np.pi + 0.4# + 0.1#2 * np.pi #np.pi+0.4

mpc = MPC(0.5,0,start_theta,0) 

states = np.zeros([200, 5])

for i_episode in range(1):
    observation = env.reset()
    state = running_state(observation)
    #env.env.state[2] = start_theta - np.pi
    action = 0
    reward_episode = 0
    num_steps = 0
    for t in range(200):
        state_var = tensor(state).unsqueeze(0).to(dtype)
        # choose mean action
        if is_disc_action:
            action = policy_net.select_action(state_var)[0].cpu().numpy()
        else:
            action = policy_net(state_var)[0][0].detach().numpy()
        # choose stochastic action
        action = int(action) if is_disc_action else action.astype(np.float64)


        #action = env.action_space.sample()
        a = mpc.update(observation[0] , observation[1], observation[2]+np.pi, observation[3], (2 * action - 1) * env.env.force_mag)
        #env.env.force_mag = abs(a) #min(100, abs(a))
        if a is None or True:
            print("Unsafe action>>> Use MPC to solve safe action")
            a = mpc.update(observation[0] , observation[1], observation[2]+np.pi, observation[3])
            if a < 0:
                action = 0
            else:
                action = 1

        states[t] = [i for i in observation[0: 4]] + [action]
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)

        next_state = running_state(observation)
        reward_episode += reward
        num_steps += 1


        env.render()
        if done or num_steps >= args.max_expert_state_num:
             break

        state = next_state


    print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

t = np.arange(200)
xs = states[:, 0]
x_dots = states[:, 1]
thetas = np.unwrap(states[:, 2])
theta_dots = states[:, 3]
actions = states[:, 4]

plt.plot(thetas, theta_dots)
plt.xlabel("Theta")
plt.ylabel("Theta_dot")
plt.title("Theta vs Theta_dot")
plt.show()


plt.plot(t, thetas)
plt.xlabel("Time")
plt.ylabel("Theta")
plt.title("Theta vs Time")
plt.show()


plt.plot(t, actions)
plt.xlabel("Time")
plt.ylabel("Actions")
plt.title("Actions vs Time")
plt.show()
