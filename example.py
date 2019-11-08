# The purpose of this file is to introduce the openai gym env API using
# a minimal example.
# More details on the API can be found here: http://gym.openai.com/docs/

import gym
env = gym.make('CartPole-v0')

print("Action Space:")
print(env.action_space)
print("Observation Space:")
print(env.observation_space)
print("Max values for observation:")
print(env.observation_space.high)
print("Min values for observation:")
print(env.observation_space.low)


print("Press enter to start sim...")
input()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Step:")
        print(observation)
        print(reward)
        print(done)
        print(info)
        print("\n")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
