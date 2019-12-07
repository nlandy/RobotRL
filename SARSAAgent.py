import numpy as np
import math
import random
from collections import defaultdict



class SARSAAgent:
    def __init__(self, env):
        self.env = env
        #self.num_actions = env.action_space.n #for cart
        self.num_actions = 4

        self.buckets = [];
        bucket_sizes = [8,8,8,8,8,8]
        temp = bucket_sizes+[2,2,2,2]
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[0]-1))
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[1]-1))
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[2]-1))
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[3]-1))
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[4]-1))
        self.buckets.append(np.linspace(0,1.5, bucket_sizes[5]-1))
        self.Q = np.zeros(tuple(bucket_sizes)+ (16,))
        print("q",self.Q[(0,0,0,0,0,0)])
        observation = env.reset()

        self.MIN_LEARNING_RATE = .1
        self.DISCOUNT_FACTOR = .9
        self.MIN_EPS = .1

    def policy(self, observation):
        #This just maps a state to an action.
        #If you learn a Q function for example, this is just max_a Q(s,a).
        #assert False, ("Error, not implemented yet.")

        state = obvservation_to_state(observation, self.buckets)
        action = get_policy(self.Q, state) # action should be some function of the observation
        return index_to_action[action]

def get_policy(Q, state):
    return np.argmax(Q[state])


def obvservation_to_state(observation, buckets):
    dict_key = []
    obs = []
    obs.append(observation['observation'][0])
    obs.append(observation['observation'][1])
    obs.append(observation['observation'][2])
    obs.append(observation['desired_goal'][0])
    obs.append(observation['desired_goal'][1])
    obs.append(observation['desired_goal'][2])

    for i in range(0, len(obs)):
        dict_key.append(np.digitize(obs[i], buckets[i]))
    return tuple(dict_key)

index_to_action = {}

index_to_action[0] = [-.5,-.5,-.5,-.5]
index_to_action[1] = [-.5,-.5,-.5,.5]
index_to_action[2] = [-.5,-.5,.5,-.5]
index_to_action[3] = [-.5,-.5,.5,.5]
index_to_action[4] = [-.5,.5,-.5,-.5]
index_to_action[5] = [-.5,.5,-.5,.5]
index_to_action[6] = [-.5,.5,.5,-.5]
index_to_action[7] = [-.5,.5,.5,.5]
index_to_action[8] = [.5,-.5,-.5,-.5]
index_to_action[9] = [.5,-.5,-.5,.5]
index_to_action[10] = [.5,-.5,.5,-.5]
index_to_action[11] = [.5,-.5,.5,.5]
index_to_action[12] = [.5,.5,-.5,-.5]
index_to_action[13] = [.5,.5,-.5,.5]
index_to_action[14] = [.5,.5,.5,-.5]
index_to_action[15] = [.5,.5,.5,.5]


action_to_index = {}
action_to_index[-1,-1,-1,-1] = 0
action_to_index[-1,-1,-1,1] = 1
action_to_index[-1,-1,1,-1] = 2
action_to_index[-1,-1,1,1] = 3
action_to_index[-1,1,-1,-1] = 4
action_to_index[-1,1,-1,1] = 5
action_to_index[-1,1,1,-1] = 6
action_to_index[-1,1,1,1] = 7
action_to_index[1,-1,-1,-1] = 8
action_to_index[1,-1,-1,1] = 9
action_to_index[1,-1,1,-1] = 10
action_to_index[1,-1,1,1] = 11
action_to_index[1,1,-1,-1] = 12
action_to_index[1,1,-1,1] = 13
action_to_index[1,1,1,-1] = 14
action_to_index[1,1,1,1] = 15

def get_actions(Q, EPS, state, env):
    if(random.random() < EPS):
        act= env.action_space.sample()
        for i in range(0,len(act)):
            if(act[i] < 0):
                act[i]=-1
            else:
                act[i]=1
        act=act.astype(int)
        index=action_to_index[tuple(act)]
    else:
        index = np.argmax(Q[state])
        act=index_to_action[index]

    return act, index


def teachSARSAAgent(env):
    agent = SARSAAgent(env)
    #agent.LEARNING_RATE = get_alpha(agent, 0)
    agent.LEARNING_RATE, startingRate = 1.0, 1.0
    agent.EPS, startingEPS = 1.0, 1.0
    num_episodes = 20000
    total_reward = np.zeros((num_episodes))
    reward_improvement = np.zeros((num_episodes))
    for i_episode in range(num_episodes):
        if(i_episode %100 == 0):
            print("episode: " + str(i_episode))
            print(agent.EPS)

        observation = env.reset()
        current_state = obvservation_to_state(observation, agent.buckets)
        current_input, current_action = get_actions(agent.Q, agent.EPS, current_state, env)
        done = False;
        i=0
        while not done:
            #env.render()
            observation, reward, done, info = env.step(current_input)

            total_reward[i_episode] += reward*agent.DISCOUNT_FACTOR**i
            if(i==0):
                start_reward=reward
            elif(i==50 or done):
                end_reward=reward
                reward_improvement[i_episode] = end_reward - start_reward

            i+=1


            next_state = obvservation_to_state(observation, agent.buckets)
            next_input, next_action = get_actions(agent.Q, agent.EPS, next_state, env)
            #agent.Q[current_state][current_action] += agent.LEARNING_RATE * (reward + agent.DISCOUNT_FACTOR * np.max(agent.Q[next_state]) - agent.Q[current_state][current_action])
            if(not done):
                agent.Q[current_state][current_action] += agent.LEARNING_RATE * (reward + agent.DISCOUNT_FACTOR * agent.Q[next_state][next_action] - agent.Q[current_state][current_action])
            if(done):
                agent.Q[current_state][current_action] += agent.LEARNING_RATE * (reward + agent.DISCOUNT_FACTOR * np.max(agent.Q[next_state]) - agent.Q[current_state][current_action])
            agent.Q[current_state][current_action] += agent.LEARNING_RATE * (reward + agent.DISCOUNT_FACTOR * np.max(agent.Q[next_state]) - agent.Q[current_state][current_action])
            current_state = next_state
            current_action = next_action
            current_input = next_input

        agent.LEARNING_RATE = max(agent.MIN_LEARNING_RATE, startingRate * math.exp(-0.00009*i_episode))
        agent.EPS = max(agent.MIN_EPS, startingEPS * math.exp(-0.00008*i_episode))


    np.save("total_reward_sarsa2_fetchreach", total_reward)
    print(total_reward[0])
    print(total_reward[100])


    np.save("reward_improvement_sarsa2_fetchreach", reward_improvement)
    #
    # Do whatever you need to do here to teach the agent.
    # When this function returns, we will not teach the agent anymore
    # and will just run its policy function.
    #
    return agent
