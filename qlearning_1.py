import gym 
import numpy as np

env=gym.make("MountainCar-v0")


#print(env.observation_space.high) #[0.6,0.07] # it provides the high value of all of the observations
#print(env.observation_space.low) # [-1.2,-0.07] this is the low
#print(env.action_space.n) # 3  it shows how many actions can we take 

LEARNING_RATE= 0.1   # it is similar as we use in deep learning
DISCOUNT = 0.95 	# It is kind of weight i.e, how important do we find the future actions 
# Discount is a kind of future action or future reward over current reward type thing.
# Agent always looks for the max Q value. Now thw current max Q value is looking for the future max Q values as well

# basically it deciedes we want to pick rewards for a short time basis or go for the long run

EPISODES=25000 # Epochs

SHOW_EVERY=5000 #after every 2000 episodes it shows its status.

#We try to gather some information about the environment


#DISCRETE_OS_SIZE=[20,20]   # This is the size of the Q matrix 

# This is never fixed this keeps changing because its necessary to change them.


DISCRETE_OS_SIZE=[20] * len(env.observation_space.high) #20*2=40 - Discrete Obseravtion space size

# This is done because environment spaces have different sizes. here 
# it is 2, it can be 5, 10 or anything depending on the environment

# So we want to seperate the range [-1.2, -0.07] to [0.6, 0.07] into 20 chunks or buckets or discrete values.
# To avoid the continous values.

# Now we try to devise how big can the step be from the lower limit to the upper limit

discrete_os_win_size=(env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#print(discrete_os_win_size)   #[0.09  0.007] 

# the first index is the space chunks and second is the velocity chunks

# Sometimes we need to change the 20 value.


epsilon = 0.5  # it is value between 0 and 1. it is the measure of the random action

# More the epsilon more is the exploratory action. But after sometime we will want our agent to stop exploring
# But some environment requires it.

START_EPSILON_DECAYING=1
END_EPSILON_DECAYING= EPISODES // 2

# Double divison gives integer

epsilon_decay_value=epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n])) # Initializing the Q Table.

# the low and high values need to be changed with the environments.
# Our rewards are mostly -1 so its good to initialize the q_table in the negatives.
# This creates a table with all of the combinations of environment variables and 3 actions for each of them
# Dimension 20 x 20 x 3 and inside each we wil have the random q values.
# Then these values will get changed using the formula.

def get_discrete_state(state):
	discrete_state=(state-env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

# As we recieve all continous values this fucntion will help us to generate discrete values 
# from the continous values. They come in tuple format. Tuple of the combination

for  episode in range(EPISODES): # creating the training phase
	
	if episode % SHOW_EVERY == 0:  # Every 2000 episodes it shows the status
		print(episode)
		render = True
	else:
		render = False

	discrete_state=get_discrete_state(env.reset())


	# env.reset() returns the initial state to us.
	# so we can directly.
	#print(discrete_state)   #( 8 10)  This is the new initial discrete step




	done=False

	while not done: #this whole thing runs only once from here so we have to keep it running and we need a episode loop
		

		if np.random.random() > epsilon:
			# np.random.random() gives a random float between 0 and 1
			# if it is greater than epsilon  go for the max one
			action = np.argmax(q_table[discrete_state]) 
		else:
			# else this gives a integer between 0 and the number of actions in the actionspace here 3
			# and choses a random action for exploring.
			action = np.random.randint(0,env.action_space.n)



		# it checks the three actions at the discrete step and finds the index of the maximum valued one

		new_state,reward,done,_ =env.step(action)

		new_discrete_state=get_discrete_state(new_state)
		# It discretizes the new step obtained by the env.step function


		#print(reward,new_state)   #we get like 1.0 [-0.17424493 -0.00265969]

		# Reward is always a -1 initially it will be -1 until we reach the flag.
		if render:
			env.render()
		if not done:
			# if complete we do not want do anything
			max_future_q=np.max(q_table[new_discrete_state])
			# the max_future_q is useful for the formula so we just need the value at the state after the action is taken
			# like if (1,5) where 1 is current state 5 is next achieved after taking some action at 1. We try to obtain the max q value at 5.

			# this actually works on back propagation. The q value gets back propagated 
			current_q=q_table[discrete_state+(action,)]
			# this is the current q value like the discrete state is somewhere in the 20 X 20 and action is 0,1 or 2 which helps to get the 
			# particular cell. because only q_table[discrete_state] will provide 3 values

			new_q= (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward * DISCOUNT * max_future_q)

			#We get a reward if we achieve the final state else its always 1

			q_table[discrete_state+(action,)]=new_q
			#updating the current state q value and the cell corresponding to the action taken with the new q value obtained from the action taken
			# We update the value after we took the action on the step

		elif new_state[0] >= env.goal_position:
			print(f"We made it by episode {episode}")
			q_table[discrete_state+(action,)]=0
			# 0 is the reward for completing things. 
			# We check whether our current postion is greater or equal to the goal state position

		discrete_state=new_discrete_state
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value # decaying the epsilon value 

	# keeping the loop running

env.close()



