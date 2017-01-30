import numpy as np

class Environment(object):    
    
    def __init__(self, arms):
        self.arms = arms
        
    def m(self):
        return len(self.arms)
        
    def pull(self, i, *ctx):
        r = self.arms[i].pull(*ctx)
        return max(a.expected(*ctx) for a in self.arms), r

class Arm(object):
    
    def pull(self):
        raise NotImplementedError

    def expected(self):
        raise NotImplementedError

class BaseArm(Arm):
    
    def __init__(self, D):
        self.D = D
    
    def pull(self):
        return self.D.rvs()
    
    def expected(self):
        return self.D.mean()

class Agent(object):

    def __init__(self, env):
        self.env = env

    def play(self, T, strat):
        m = self.env.m()
        Q = np.zeros(m)
        N = np.zeros(m)
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        for t in xrange(T):
            i = strat.play(Q=Q, N=N, m=m, t=t) 
            opt, reward = self.env.pull(i)
            N[i] += 1
            Q[i] += (reward - Q[i]) / N[i]
            choices[t] = i
            opt_rewards[t, ]  = np.array([opt, reward])
        return choices, opt_rewards
