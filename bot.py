import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter 

# learning rate
lr = 1e-3

#environment
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 50
intial_games = 10000

def random_games():
	#Each of this is its own game
	for episode in range(5):
		env.reset()
		#this is each frame upto 200
		for t in range(goal_steps):
			#displaying the environment
			#takes much longer to display it
			env.render()
			#action to be made i.e., move left if 0 or move right if 1
			action = env.action_space.sample()
			#executing the environment with the above action
			observation,reward,done,info = env.step(action)
			if done:
				break


def initial_population():
	#[observation,moves]
	training_data = []
	#all scores
	scores = []
	#just the scores that met our threshold
	accepted_scores = []
	#Iterating through no of games
	for _ in range(intial_games):
		score = 0
		#move specifically from this environment:
		game_memory = []
		#previous observation that we saw
		prev_observation = []
		#for each frame in 200
		for _ in range(goal_steps):
			#choose action i.e., move left or move right
			action = random.randrange(0,2)
			#do it
			observation,reward,done,info = env.step(action)

			#storing previous observation
			if len(prev_observation) > 0 :
				game_memory.append([prev_observation,action])

			prev_observation = observation
			score += reward
			if done:
				break

		#if our score is higher than threshold then we will like to save every move
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				#converting to one-hot
				#this is the output layer for our neyral network
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

			#saving our training data
			training_data.append([data[0],output])

		#reset to play again
		env.reset()

		#save all overall scores
		scores.append(score)

	#saving training data for later refernces
	training_data_save = np.array(training_data)
	np.save('saved.npy',training_data_save)

	#stats to further illustrate our neural network magic
	print('Average accepted scores:',mean(accepted_scores))
	print('Median score for accepted scores:',median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data

