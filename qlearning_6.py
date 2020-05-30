#Design a class for an agent using a convnet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import keras.backend.tensorflow_backend as backend
import time
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import tensorflow as tf


REPLAY_MEMORY_SIZE=50000
MIN_REPLAY_MEMORY_SIZE=1000
MODEL_NAME="256x2" # it is the neural network
MINIBATCH_SIZE=64
DISCOUNT=0.99
UPDATE_TARGET_EVERY=5

##tensorboard class model
# this is done because the keras and the tensorflow wants to create
# the tensorboard every time we .fit the model but we really don't want that 
# normally we don't need this because we fit once but here we need to fit constantly.
# So,we need it. 

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

## this is the main blob class we created
class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):     # THIS IS TO CHECK WHETHER TWO BLOBS ARE OVER EACH OTHER
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

     # the choices have been increased lateral and top down movement added  and no movement added
    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


##This is the previous environment only in a Object oriented manner
class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
    #This is to withdraw an actual image

env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:

	def __init__(self):

		#prepared main model # it is trained
		self.model= self.create_model()

		#target model     # it is used for prediction
		self.target_model= self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		#We have two models beacuse in reinforcement learning the model will try out new explorations
		#or some times go for the best action depending on the value of epsilon.
		#So, it is fit a number of times but if we keep changing our real target model and let it train
		#and predict all the time it will be very hard to keep any sort of consistency
		# So, we will use two models, one that will train and other that will set the weights from the 
		#trained one and predict keeping a sense of consistency. There will be a fit for multuple time 

		self.replay_memory=deque(maxlen=REPLAY_MEMORY_SIZE)
		#  Deque is preferred over list in the cases where we need quicker append and pop operations 
		#from both the ends of container, as deque provides an O(1) time complexity for append and pop operations 
		#as compared to list which provides O(n) time complexity.
		#So this kind of replaces the list

		#self.model gets .fit in every single step. Normally we use the fit method in batches but here it will be 
		#fit to a single value

		#Now the thing is when we train a batch of 64 samples it trains for 64 samples together and so doesnot overfit 
		#to any particular sample. But here it is trained or fit using a particular sample. So there is a chance that 
		# the model will be overfit accordinn\g to that particular sample at that instance

		#This also needs to be taken care of.Now we do this using this deque method.
		#We pickup random samples from the 50,000 steps and create a batch which we use to train to lower the impact 
		#of training on a single sample data.

		#This way we try to smoothen both the areas. Firstly we use a batch which smoothens the learning and cuts out the 
		#chance of overfitting due to single data.
		#We remove the problem with consistency in prediction using  a seperate model which will be the actual target model and 
		#training the data on a seperate model. We update the target model after a period or a term of episodes.

		self.tensorboard= ModifiedTensorBoard(log_dir= f"logs/{MODEL_NAME}-{int(time.time())}")

		self.target_update_counter=0 
		# This is used to internally track when to update the model

	
	def create_model(self):
		model=Sequential()
		model.add(Conv2D(256,(3, 3),input_shape=env.OBSERVATION_SPACE_VALUES))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2, 2))
		model.add(Dropout(0.2))

		model.add(Conv2D(256,(3, 3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2, 2))
		model.add(Dropout(0.2))

		model.add(Flatten())
		#this converts the whole thing into an array so that we can pass this 
		#through the dense layers
		model.add(Dense(64))
		model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
		# This is the output layer with linear activation

		model.compile(loss="mse",optimizer=Adam(lr=0.001),metrics=['accuracy'])
		return model

		# We can see the size of the output is equal to the action space size and the input size is 
		# equal to the observation space values. So, it is evident that we pass the observation feature space
		# as the input and the model gives us the action we need to perform here based on that observation
	
	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	## The transition contains of the observation space, action,reward
	# new observation space and the confirmation about whether or not the 
	# task was complete. It is done so that we can use the formula

	def get_qs(self,state):
		return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

	#It is just unpacking the prediction. It comes as a list of one value so we use
	#the [0] and /255 is done to scale as it is a RGB image



	def train(self, terminal_state, step):

		# Here we will create a batch to train our model. 
		# we will randomly pick element to create a batch
		# now the thing is normally we have batches of 32 and 64 but 
		# those will be very small for a memory size of 50,000 So we will be using a 
		# batch size of 1000
		if len(self.replay_memory)<MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)

		# here the neural net handles the learning rate and all those
		# but we still need the reward, discount and the future value

		current_states=np.array([transition[0] for transition in minibatch])/255

		# it creates the list of all the states or observations
		# now again they are image data so have values from 1 to 255 so we divide by 255 to scale the values 
		# between 0 and 1

		current_qs_list=self.model.predict(current_states)

		new_current_states=np.array([transition[3] for transition in minibatch])/255
		# these current states are produced when we take action on current states and it is predicted by our rough model
		# this is equivalent to the new obs

		future_qs_list=self.target_model.predict(new_current_states)
		# this is the future q list created from our stable model.
		# it is for the q values formula

		X=[]		#feature set-images from the game
		Y=[]		#action set


		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch): 
			# this is the transition tuples from where the transition[indexes] are coming
			if not done:
				max_future_q= np.max(future_qs_list[index])
				new_q=reward + DISCOUNT * max_future_q

			else:

				new_q=reward


			current_qs = current_qs_list[index]
			current_qs[action]=new_q

			#now here we are assigning the new_q value to the current_qs[acion]
			#current_qs_list contains a list of list of actions. i,e if there are 4 actions 
			# it has a list of 4 actions for 64 current states.
			# So it selects the index we are currently working on and assigns the q value to that 
			# particular action

			X.append(current_state)
			Y.append(current_qs)

		self.model.fit(np.array(X)/255,np.array(Y),batch_size=MINIBATCH_SIZE,
			verbose=0, callbacks=[self.tensorboard] if terminal_state else None)

		#We will fit these if we are on our terminal state or we will fit nothing

		# updating to determine if we want to update our target model just yet
		if terminal_state:
			self.target_update_counter+=1


		if self.target_update_counter> UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter=0


agent=DQNAgent()


for episode in tqdm(range(1,EPISODES+1),ascii=True,unit= "episode"):
	agent.tensorboard.step=episode


	episode_reward= 0
	step=1
	current_state =env.reset()


	done= False

	while not done:
		if np.random.random() > epsilon:
			action=np.argmax(agent.get_qs(current_state))
			
		else:
			action=	np.random.randint(0,env.ACTION_SPACE_SIZE)


		new_state, reward, done= env.step(action)

		episode_reward+=reward

		if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
			env.render()

		agent.update_replay_memory((current_state, action, reward, new_state, done))
		agent.train(done,step)

		current_state = new_state

		step+=1
	# Append episode reward to a list and log stats (every given number of episodes)
	ep_rewards.append(episode_reward)
	if not episode % AGGREGATE_STATS_EVERY or episode == 1:
		average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
		min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
		max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
		agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
		
		# Save model, but only when min reward is greater or equal a set value
		if min_reward >= MIN_REWARD:
			agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        #if the model scores good save the model
        # if it hits the enemy it is negative 200
        # if it doesnot hit the enemy and doesnot reach the food it is somewhat 
        #positive, so save the model.
    # Decay epsilon
	if epsilon > MIN_EPSILON:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)










