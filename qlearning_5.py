#Design a class for an agent using a convnet
from keras.models import sequential
from keras.layers import Dense, Dropout, Conv2D, Maxpooling2D, Activation, Flatten
from keras.callback import Tensorboard
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np
import random


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
		model=sequential()
		model.add(Conv2D(256,(3, 3),input_shape=env.OBSERVATION_SPACE_VALUES))
		model.add(Activation("relu"))
		model.add(Maxpooling2D(2, 2))
		model.add(Dropout(0.2))

		model=sequential()
		model.add(Conv2D(256,(3, 3)))
		model.add(Activation("relu"))
		model.add(Maxpooling2D(2, 2))
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

	def get_qs(self,state,step):
		return self.model_predict(np.array(state).reshape(-1,*state.shape)/255)[0]

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

				new_q-reward


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








