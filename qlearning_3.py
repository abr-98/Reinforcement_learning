#Making own environments

#this has a food, a player and an enemy. Player can move enemy does not move. The
#food is the target. As long as we don't hit the enemy its fine. Then we define the actions

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time #used to set dynamic q_table file names

style.use("ggplot")

SIZE=10
#It will be a 10 x 10 grid. The enemy, the food and the player will be initialized at random
#location of the 10 x 10 grid

#We need to take care of the size because the whole thing is dependent on both this size and the size of
#the action space. Both of them decide the size of the q_table. As they increase exponentially

HM_EPISODES=25000
MOVE_PENALTY=1
ENEMY_PENALTY=300 #It is the penalty in case if we hit the enemy

#If we hit this amount will be substracted

FOOD_REWARD=25

epsilon=0.9
EPS_DECAY=0.9998

SHOW_EVERY=3000

start_q_table= None # or a filename if we have an existing one


LEARNING_RATE= 0.1
DISCOUNT=0.95

PLAYER_N=1
FOOD_N=2
ENEMY_N=3

# These are kinda codes to be defined in the colour dictionary

d = {1: (255,175,0),
	2: (0,255,0),
	3: (0,0,255)}

#BGR
# Player is greenisg blue
# Food is green
# Enemy is red

#BLOB CLASS

#if we try to pass absolute location as observation, the observation space
#will be really big. So, our observation is relative position of the enemy and 
# food wrt the player.


class Blob:
	def __init__(self):
		self.x=np.random.randint(0,SIZE)
		self.y=np.random.randint(0,SIZE)

	#this will give the random (x,y) pairs on the grid.There will be some problems
	#if any of our variables spawn on each other.

	def __str__(self):
		return f"{self.x},{self.y}"
	#Debugger: It prints the current values of X and Y

	def __sub__(self,other):
		return (self.x-other.x, self.y-other.y)

	# it substracts one blob from another

	def action(self,choice):
		if choice == 0:
			self.move(x=1,y=1)
		if choice == 1:
			self.move(x=-1,y=-1)
		if choice == 2:
			self.move(x=-1,y=1)
		if choice == 3:
			self.move(x=1,y=-1)

	#this is the action space of the environment. To keep the space small we are just 
	#allowing the player to move diagonally. And also as the player will only move so 
	# only self id passed here. 
	# It is very hard for q learning to use these movements to reach the food because if 
	#it is spawned in the odd and food is in the even. It will need to flip its position to 
	#diagonally reach it using the walls. This is pretty hard for am agent to discover


	def move(self, x=False, y=False):
		if not x:
			self.x+=np.random.randint(-1,2)

			#it has the ability to generate values (-1,0,1) if it is not provided any value

		else: 
			self.x+=x

		if not y:
			self.y+=np.random.randint(-1,2)

			#it has the ability to generate values (-1,0,1) if it is not provided any value

		else: 
			self.y+=y

		if self.x <0:
			self.x = 0

		elif self.x >SIZE - 1:
			self.x=SIZE-1

		if self.y <0:
			self.y = 0

		elif self.y >SIZE - 1:
			self.y=SIZE-1

		# this is done to restrict the movement if the agent hits the wall then it can move in a top down
		# bottom up and side ways manner

if start_q_table is None:
	q_table= {}
	
	#Now we are using a relative observation space i.e, our observation space is (x1,y1),(x2,y2)
	#where (x1,y1) is the relative difference between the player and the food and the (x2,y2) is the 
	#difference between the player and the enemy which we will obtain using the substract operation
	# Now we will iterate through them so four loops.

	for x1 in range(-SIZE+1,SIZE):	#SIZE x SIZE actually -SIZE+1 to SIZE-1
		for y1 in range(-SIZE+1,SIZE):
			for x2 in range(-SIZE+1,SIZE):
				for y2 in range(-SIZE+1,SIZE):
					q_table[((x1,y1),(x2,y2))]= [np.random.uniform(-5,0) for i in range(4)]
					# The key is tuple of a tuple 
					# we initialize this with uniform random values from -5 to 0. and 4 times becase 4 values.

else:
	with open(start_q_table,"rb") as f:
		q_table=pickle.load(f)

episode_rewards=[]

for episode in range(HM_EPISODES):
	player=Blob()
	food=Blob()
	enemy=Blob()


	if episode % SHOW_EVERY == 0:
		print(f"on # {episode}, episode: {epsilon}")
		print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
		show=True
	else:
		show=False


	episode_reward=0
	for i in range(200):
		obs=(player-food,player-enemy)
		# Creating the observation list

		if np.random.random() >epsilon:
			action=np.argmax(q_table[obs])
		else:
			action=np.random.randint(0,4)
		#this again is producing some degree of randomness to our agent. If 
		#the value grater then epsilon it will go with the action that produce greatest_value 
		#else it will select a random action in order to explore

		player.action(action)
		'''
		# Maybe later incorporate these

		enemy.move()
		food.move()
		'''

		#Reward Process

		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY

		elif player.x == food.x and player.y == food.y:
			reward= FOOD_REWARD

		else: 
			reward = -MOVE_PENALTY

		#if it strikes nothing it will have the mve penalty as -1

		new_obs=(player-food,player-enemy)
		# going for the new state
		max_future_q=np.max(q_table[new_obs])

		# for the formula

		current_q= q_table[obs][action]

		# action because always we are not going to use the best action

		if reward == FOOD_REWARD:
			new_q=FOOD_REWARD

		elif reward == -ENEMY_PENALTY:
			new_q=-ENEMY_PENALTY

		else:
			new_q=(1- LEARNING_RATE) * current_q+ LEARNING_RATE *(reward + DISCOUNT * max_future_q)

		q_table[obs][action]=new_q

		if show:
			env =np.zeros((SIZE, SIZE,3),dtype=np.int8)

			env[food.y][food.x] = d[FOOD_N]
			env[player.y][player.x] = d[PLAYER_N]
			env[enemy.y][enemy.x] = d[ENEMY_N]

			## Changing the grid colour.

			Img= Image.fromarray(env, "RGB")
			Img=Img.resize((300,300))
			cv2.imshow("",np.array(Img))
			if reward ==  FOOD_REWARD or reward == -ENEMY_PENALTY:
				if cv2.waitKey(500) & 0xFF== ord("q"):
					break
			else:
				if cv2.waitKey(1) & 0xFF== ord("q"):
					break	

		episode_reward+=reward
		if reward ==  FOOD_REWARD or reward == -ENEMY_PENALTY:
			break

	episode_rewards.append(episode_reward)
	epsilon*= EPS_DECAY

moving_avg=np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
## It provides a moving average image

plt.plot([i for i in range(len(moving_avg))],moving_avg)

plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"q_table-{int(time.time(_))}.pickle","wb") as f:
	pickle.dump(q_table,f)



