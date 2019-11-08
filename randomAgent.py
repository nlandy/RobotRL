class randomAgent:
    def __init__(self, env):
        self.env = env
    def policy(self, observation):
        action = self.env.action_space.sample()
        return action

def teachRandomAgent(env):
    agent = randomAgent(env)
    return agent
