import numpy as np

class Agent(object):
    
    def play(self, T):
        raise NotImplementedError

class RandomAgent(Agent):
    
    def __init__(self, env):
        self.env = env

    def play(self, T):
        m = self.env.m()
        choices = np.random.randint(0, high=m, size=T)
        rewards_opt = np.array([self.env.pull(i) for i in choices])
        return choices, rewards_opt

class EpsilonGreedyAgent(Agent):
    
    def __init__(self, env, e):
        self.env = env
        self.e = e
    
    def play(self, T):
        Q = np.zeros(self.env.m())
        N = np.zeros(self.env.m())
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        for t in xrange(T):
            i = np.random.randint(0, high=self.env.m()) if self.e > np.random.rand() else np.argmax(Q)
            opt, reward = self.env.pull(i)
            N[i] += 1
            Q[i] += (reward-Q[i]) / N[i]
            choices[t] = i
            opt_rewards[t, ]  = np.array([opt, reward])
        return choices, opt_rewards

