class BAMDPAgent:
    def __init__(self, env):
        self.env = env
    def BAMDPPolicy(self, observation):
        assert False, ("Error, not implemented yet.")
        action = None # action should be some function of the observation
        return action

def BAMDPLearn(env):
    agent = BAMDPAgent(env)
    return agent.BAMDPPolicy
