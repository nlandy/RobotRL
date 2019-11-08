# This is the main file for our project.
# To run it, just run "python main.py".
# Make sure to set ALGO to which algorithm you want to try.

import gym
from QLearnAgent import QLearn
from SARSAAgent import SARSALearn
from BAMDPAgent import BAMDPLearn
from randomAgent import randLearn

env_name = 'CartPole-v0'

ALGO = "Q-Learning"
# ALGO = "SARSA"
# ALGO = "BAMDP"
# ALGO = "Random"

ALGO_DICT = {
"Q-Learning" : QLearn,
"SARSA" : SARSALearn,
"BAMDP" : BAMDPLearn,
"Random" : randLearn
}

def main():
    env = gym.make(env_name)
    if(ALGO in ALGO_DICT.keys()):
        policy = ALGO_DICT[ALGO](env)
    else:
        assert False, ("Algorithm not implemented and included yet.")

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = policy(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

if __name__== "__main__":
  main()
