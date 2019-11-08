class randomAgent:
    def __init__(self, env):
        self.env = env
    def randPolicy(self, observation):
        action = self.env.action_space.sample()
        return action

def randLearn(env):
    agent = randomAgent(env)
    return agent.randPolicy
