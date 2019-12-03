import numpy as np
import time
import matplotlib.pyplot as plt
import math

class MLEMBAgent:
    def __init__(self, env):
        self.env = env
        self.statesize = 8
        self.actionsize = 8
        self.N = np.zeros((8,8,8)) #s, a, s'
        self.rho = np.zeros((8,8)) #s, a
        #self.Q = np.zeros((8,8)) #s, a
        self.Q = np.asarray([[-14.12857193, -14.12958728, -14.12970078, -14.13124415, -14.13161919, -14.1322074,  -14.13165685, -14.13313277],
                             [-14.18573099, -14.18413926, -14.18715815, -14.18486409, -14.18726325, -14.18514344, -14.1881218,  -14.18638078],
                             [-14.15988371, -14.15961278, -14.1581087,  -14.15879991, -14.1617476, -14.1621839,  -14.15976774, -14.1598562 ],
                             [-14.21679582, -14.21542602, -14.21430797, -14.21319355, -14.21782875, -14.21541015, -14.21716419, -14.21405064],
                             [-14.11268177, -14.11400332, -14.11548796, -14.11536908, -14.11272027, -14.11186244, -14.11448915, -14.11393612],
                             [-14.214822,   -14.2114045,  -14.2160697,  -14.21348614, -14.21281588, -14.21057885, -14.21442619, -14.21203492],
                             [-14.12286976, -14.12316139, -14.12050998, -14.1203347,  -14.12110246, -14.12199611, -14.11846268, -14.12007209],
                             [-14.18149024, -14.17857618, -14.18014624, -14.17767006, -14.17956494, -14.17654888, -14.17871283, -14.17587644]], dtype=np.float32)
        self.gamma = 0.99
        self.eps = 0.999999

    def policy(self, observation):
        obs = self.mapObservation(observation)
        bins = np.linspace(-1.0, 1.0, num=10)
        #actNum = np.argmax(self.Q[obs], axis=-1)
        actNum = np.random.choice(np.where(self.Q[obs] == self.Q[obs].max())[0])
        action = self.unmapAction(actNum, bins)

        return action

    def learningUpdate(self, oldObservation):
        action = self.chooseAction(oldObservation)
        newObservation, reward, done, info = self.env.step(action)
        self.updateEstimates(oldObservation, newObservation, action, reward)
        return newObservation, reward, done, info

    def updateEstimates(self, oldObservation, newObservation, action, reward):
        oldObs = self.mapObservation(oldObservation)
        newObs = self.mapObservation(newObservation)
        act = self.mapAction(action)
        #print(action, act)
        self.N[oldObs, act, newObs] += 1
        self.rho[oldObs, act] += reward
        R = self.rho[oldObs, act]/np.sum(self.N[oldObs, act])
        T = self.N[oldObs, act]/np.sum(self.N[oldObs, act])
        self.Q[oldObs, act] = R + self.gamma * np.sum(T*np.max(self.Q, axis=-1))

    def mapObservation(self, observation):
        desired_goal = observation['desired_goal']
        observ = observation['observation']
        state = desired_goal - observ[0:3]
        #state = observation

        #print(state)

        bins0 = np.linspace(0, 1, num=2)
        bins1 = np.linspace(0, 1, num=2)
        bins2 = np.linspace(0, 1, num=2)
        #bins3 = np.linspace(-0.9, 0.9, num=12)

        digitState = np.zeros(3, dtype=int)
        digitState[0] = int(np.digitize(state[0], bins0[0:-1]))
        digitState[1] = int(np.digitize(state[1], bins1[0:-1]))
        digitState[2] = int(np.digitize(state[2], bins2[0:-1]))
        #digitState[3] = int(np.digitize(state[3], bins3[0:-1]))

        #print(digitState)

        #vals = bins[digitState]
        #index = int(vals[0] + vals[1]*10 + vals[2]*100 + vals[3]*1000)
        index = np.ravel_multi_index(tuple(digitState), (2,2,2))
        #print(digitState, index)
        return index

    def mapAction(self, action):
        bins = np.linspace(-0.049, 0.05, num=2)
        digitState = np.zeros(3, dtype=int)
        digitState[0] = int(np.digitize(action[0], bins[0:-1]))
        digitState[1] = int(np.digitize(action[1], bins[0:-1]))
        digitState[2] = int(np.digitize(action[2], bins[0:-1]))
        index = np.ravel_multi_index(tuple(digitState), (2,2,2))
        return index

    def unmapAction(self, actNum, bins):
        bins = np.linspace(-0.049, 0.05, num=2)
        action = np.zeros(3, dtype=np.uint8)
        action[0], action[1], action[2] = np.unravel_index(actNum, (2,2,2))
        action = np.append(bins[action], 0)
        return action

    def chooseAction(self, observation):
        #if(np.random.uniform() < self.eps):
        if True:
            bins = np.linspace(-0.05, 0.05, num=2)
            return np.append(np.random.choice(bins, 3), 0)
        else:
            return self.policy(observation)


def teachMLEMBAgent(env):
    startTime = time.time()
    agent = MLEMBAgent(env)
    num_episodes = 10000
    total_reward = np.zeros((num_episodes))
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/350)))
    for i_episode in range(num_episodes):
        agent.eps = get_epsilon(i_episode)
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
    print(agent.Q.tolist())
    print(np.argmax(agent.Q, axis=-1))
    print("Press enter to continue to sim...")
    input()
    return agent
