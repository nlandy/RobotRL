class BAMDPAgent:
    def __init__(self, env):
        self.env = env
    def policy(self, observation):
        assert False, ("Error, not implemented yet.")
        action = None # action should be some function of the observation
        return action

def teachBAMDPAgent(env):
    agent = BAMDPAgent(env)
    return agent
