import numpy as np
import time
import matplotlib.pyplot as plt
import math
from dqn_agent import Agent
import torch

class DQNAgent:
    def __init__(self, env):
        self.env = env
        #self.statesize = 13
        self.statesize = 4
        #self.actionsize = 8
        self.actionsize = 2
        self.dqn = Agent(self.statesize, self.actionsize, 42)
        self.gamma = 0.99
        self.eps = 0.999999

    def policy(self, observation):
        #observation = np.concatenate((np.asarray(observation['observation'],dtype=np.float64), np.asarray(observation['desired_goal'],dtype=np.float64)))
        actNum = self.dqn.act(observation)
        action = self.unmapAction(actNum)
        return action

    def learningUpdate(self, oldObservation):
        action = self.chooseAction(oldObservation)
        newObservation, reward, done, info = self.env.step(action)
        #pass_oldObservation = np.concatenate((np.asarray(oldObservation['observation'],dtype=np.float64), np.asarray(oldObservation['desired_goal'],dtype=np.float64)))
        #pass_newObservation = np.concatenate((np.asarray(newObservation['observation'],dtype=np.float64), np.asarray(newObservation['desired_goal'],dtype=np.float64)))
        pass_oldObservation = np.asarray(oldObservation, dtype=np.float64)
        pass_newObservation = np.asarray(newObservation, dtype=np.float64)
        reward = np.asarray(reward, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        done = np.asarray(int(done), dtype=np.float64)
        self.updateEstimates(pass_oldObservation, action, reward, pass_newObservation, done)
        return newObservation, reward, done, info

    def updateEstimates(self, oldObservation, action, reward, newObservation, done):
        act = self.mapAction(action)
        oldObservation = np.expand_dims(oldObservation, axis=0)
        newObservation = np.expand_dims(newObservation, axis=0)
        act = np.expand_dims(np.expand_dims(act, axis=0), axis=0)
        reward = np.expand_dims(reward, axis=0)
        done = np.expand_dims(done, axis=0)
        experience = (torch.from_numpy(oldObservation), torch.from_numpy(act), torch.from_numpy(reward), torch.from_numpy(newObservation), torch.from_numpy(done))
        self.dqn.step(*experience)

    def mapAction(self, action):
        #bins = np.linspace(-0.049, 0.05, num=2)
        #digitState = np.zeros(3, dtype=int)
        #digitState[0] = int(np.digitize(action[0], bins[0:-1]))
        #digitState[1] = int(np.digitize(action[1], bins[0:-1]))
        #digitState[2] = int(np.digitize(action[2], bins[0:-1]))
        #index = np.ravel_multi_index(tuple(digitState), (2,2,2))
        #index = np.asarray(index)
        #return index
        return action

    def unmapAction(self, actNum):
        #bins = np.linspace(-0.049, 0.05, num=2)
        #action = np.zeros(3, dtype=np.uint8)
        #action[0], action[1], action[2] = np.unravel_index(actNum, (2,2,2))
        #action = np.append(bins[action], 0)
        #return action
        return actNum

    def chooseAction(self, observation):
        if(np.random.uniform() < self.eps):
            #bins = np.linspace(-0.05, 0.05, num=2)
            #return np.append(np.random.choice(bins, 3), 0)
            arr = int(np.random.uniform() > 0.5)
            return arr
        else:
            return self.policy(observation)


def teachDQNAgent(env):
    startTime = time.time()
    agent = DQNAgent(env)
    num_episodes = 10000
    total_reward = np.zeros((num_episodes))
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/350)))
    for i_episode in range(num_episodes):
        agent.eps = get_epsilon(i_episode)
        agent.eps = 0.99
        observation = agent.env.reset()
        for t in range(51):
            observation, reward, done, info = agent.learningUpdate(observation)
            total_reward[i_episode] += reward * agent.gamma**t
            if done:
                break
        if(i_episode % 10 == 0):
            print("Finished epsisode {}.".format(i_episode))
    print("Time learning: {}.".format(time.time()-startTime))
    plt.plot(total_reward)
    plt.ylabel('Cumulative Reward')
    plt.show()
    print("Press enter to continue to sim...")
    input()
    return agent
