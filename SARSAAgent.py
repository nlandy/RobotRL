import numpy as np
import math
from collections import defaultdict

class SARSAAgent:
    def __init__(self, env):
        self.env = env
        self.num_actions = env.action_space.n
        self.Q = defaultdict(lambda: [0.0]*self.num_actions)
        self.buckets = [];
        self.buckets.append(np.linspace(-4.8, 4.8, 15))
        self.buckets.append(np.linspace(-3, 3, 15))
        self.buckets.append(np.linspace(-.22, .22, 15))
        self.buckets.append(np.linspace(-3, 3, 15))
        observation = env.reset()
        assert len(self.buckets) == len(observation), "Error, there aren't buckets defined for every observation, buckets: "+str(len(self.buckets)) + ", observations: "  + str(len(observation))
        
        self.LEARNING_RATE = .01
        self.DISCOUNT_FACTOR = .9
        self.EPS = 1
        
    def policy(self, observation):
        #This just maps a state to an action.
        #If you learn a Q function for example, this is just max_a Q(s,a).
        #assert False, ("Error, not implemented yet.")
        
        state = obvservation_to_state(observation, self.buckets)
        action = get_actions(self.Q, state) # action should be some function of the observation
        return action



def obvservation_to_state(observation, buckets):
    dict_key = []
    for i in range(0, len(observation)):
        dict_key.append(np.digitize(observation[i], buckets[i]))
    return tuple(dict_key)

def get_actions(Q, state):
    return Q[state].index(max(Q[state]))

def teachSARSAAgent(env):
    agent = SARSAAgent(env)
    

    for i_episode in range(50000):
        agent.EPS=1/math.sqrt(i_episode+1)
        
        #print(agent.Q)
        observation = env.reset()
        current_state = obvservation_to_state(observation, agent.buckets)
        current_action = get_actions(agent.Q, current_state)
        for t in range(100):
            #env.render()
            observation, reward, done, info = env.step(current_action)
            next_state = obvservation_to_state(observation, agent.buckets)
            if np.random.uniform() < agent.EPS:
                next_action = env.action_space.sample()
            else:			
                next_action = get_actions(agent.Q, next_state)
                

            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            else:
                agent.Q[current_state][current_action] += agent.LEARNING_RATE*(reward - agent.DISCOUNT_FACTOR*(agent.Q[next_state][next_action])- agent.Q[current_state][current_action])
                current_state = next_state
                current_action = next_action
    
            
    #
    # Do whatever you need to do here to teach the agent.
    # When this function returns, we will not teach the agent anymore
    # and will just run its policy function.
    #
    return agent
