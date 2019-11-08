class QLearningAgent:
    def __init__(self, env):
        self.env = env
    def policy(self, observation):
        #This just maps a state to an action.
        #If you learn a Q function for example, this is just max_a Q(s,a).
        assert False, ("Error, not implemented yet.")
        action = None # action should be some function of the observation
        return action

def teachQLearnAgent(env):
    agent = QLearningAgent(env)
    #
    # Do whatever you need to do here to teach the agent.
    # When this function returns, we will not teach the agent anymore
    # and will just run its policy function.
    #
    return agent
