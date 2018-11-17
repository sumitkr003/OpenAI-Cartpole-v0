import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median

# learning rate
lr = 1e-3

#environment
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirements = 50
intial_game = 10000

def random_games():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observation,reward,done,info = env.step(action)
			if done:
				break

if __name__ == '__main__':
	random_games()