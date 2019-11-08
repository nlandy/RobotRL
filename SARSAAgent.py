class SARSAAgent:
    def __init__(self, env):
        self.env = env
    def SARSAPolicy(self, observation):
        assert False, ("Error, not implemented yet.")
        action = None # action should be some function of the observation
        return action

def SARSALearn(env):
    agent = SARSAAgent(env)
    return agent.SARSAPolicy
