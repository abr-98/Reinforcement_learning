import gym 
import numpy as np
import matplotlib.pyplot as plt

''' Now we cant always guess the steps. So we need to focus on the Rewards systems
For simple tasks its enough to track the rewards after a episode but for complex it is 
not enough. This tutorial shows a way we can track them'''



env=gym.make("MountainCar-v0")



LEARNING_RATE= 0.1
DISCOUNT = 0.95	
EPISODES=30000

SHOW_EVERY=2000

DISCRETE_OS_SIZE=[120] * len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


epsilon = 0.7

START_EPSILON_DECAYING=1
END_EPSILON_DECAYING= EPISODES // 2


epsilon_decay_value=epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-10,high=-5,size=(DISCRETE_OS_SIZE + [env.action_space.n])) 


ep_rewards=[]
aggr_ep_rewards= {'ep':[], 'avg': [], 'min': [], 'max':[]}
# episode rewards track the rewards for the episodes as the list
# aggregate is there to keep a record. 'ep' is kind of the x axis. As we go up avg increases.
# they are used for tracking details of every episode
# max shows the best model min shows the worst model




def get_discrete_state(state):
	discrete_state=(state-env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))



for  episode in range(EPISODES): 
	episode_reward=0
	if episode % SHOW_EVERY == 0:  
		print(episode)
		render = True
	else:
		render = False

	discrete_state=get_discrete_state(env.reset())

	done=False

	while not done: 
		

		if np.random.random() > epsilon:
			
			action = np.argmax(q_table[discrete_state]) 
		else:
			
			action = np.random.randint(0,env.action_space.n)



		new_state,reward,done,_ =env.step(action)

		episode_reward+=reward
		#It updates the rewards and stores it through a phase

		new_discrete_state=get_discrete_state(new_state)
		

		
		if render:
			env.render()
		if not done:
			
			max_future_q=np.max(q_table[new_discrete_state])
			
			current_q=q_table[discrete_state+(action,)]
			
			new_q= (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward * DISCOUNT * max_future_q)

			
			q_table[discrete_state+(action,)]=new_q
			
		elif new_state[0] >= env.goal_position:
			print(f"We made it by episode {episode}")
			q_table[discrete_state+(action,)]=0
			
		discrete_state=new_discrete_state
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value 

	ep_rewards.append(episode_reward)
	# It appends the whole reward to the list after completing one episode

	if episode % SHOW_EVERY==0:
		np.save(str(episode)+"q_table.npy",q_table)
		#We need to save th q table pretty frequently due to the fact that it has a high degree of randomness here
		#So, sometimes even after recieving great accuracy it may fall because the q_table gets modified.
		#sometimes we need to save the q_table in order to retrieve that condition
		 
		average_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))


		print(f"episode: {episode} avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}")
		# recording the values.

env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()


