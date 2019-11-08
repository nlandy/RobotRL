class QLearningAgent:
    def __init__(self, env):
        self.env = env
    def QLearningPolicy(self, observation):
        assert False, ("Error, not implemented yet.")
        action = None # action should be some function of the observation
        return action

def QLearn(env):
    agent = QLearningAgent(env)
    return agent.QLearningPolicy
