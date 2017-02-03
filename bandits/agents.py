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
        m = self.env.m()
        Q = np.zeros(m)
        N = np.zeros(m)
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        for t in xrange(T):
            i = np.random.randint(0, high=m) if self.e > np.random.rand() else np.argmax(Q)
            opt, reward = self.env.pull(i)
            N[i] += 1
            Q[i] += (reward-Q[i]) / N[i]
            choices[t] = i
            opt_rewards[t, ]  = np.array([opt, reward])
        return choices, opt_rewards

class UCB1Agent(Agent):
    
    def __init__(self, env):
        self.env = env
    
    def play(self, T):
        m = self.env.m()
        Q = np.zeros(m)
        N = np.ones(m)
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        # init
        for i in xrange(m):
            opt_rewards[i, ] = self.env.pull(i)
            choices[i] = i
            Q[i] = opt_rewards[i, 1]

        for t in xrange(m, T):
            i = np.argmax(Q + np.sqrt((2. * np.log(t+1)) / N))
            opt, reward = self.env.pull(i)
            N[i] += 1
            Q[i] += (reward-Q[i]) / N[i]
            choices[t] = i
            opt_rewards[t, ]  = np.array([opt, reward])
        return choices, opt_rewards

class ThompsonBernoulliAgent(Agent):
    
    def __init__(self, env, alpha, beta):
        self.env = env
        self.alpha = alpha
        self.beta = beta
    
    def play(self, T):
        from scipy.stats import beta

        m = self.env.m()
        S = np.array([[self.alpha, self.beta]] * m)
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        for t in xrange(T):
            i = np.argmax([beta(*S[i]).rvs() for i in xrange(m)])
            opt, reward = self.env.pull(i)
            S[i, reward ^ 1] += 1
            choices[i] += 1
            opt_rewards[t, ] = np.array([opt, reward])
        return choices, opt_rewards    

class Exp3Agent(Agent):

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        
    def play(self, T):
        from scipy.stats import rv_discrete
        
        m = self.env.m()
        actions = [i for i in xrange(m)]
        weights = np.ones(m)
        choices = np.zeros(T)
        opt_rewards = np.zeros((T, 2))
        for t in xrange(T):
            p = (1 - self.gamma) * weights / np.sum(weights) + self.gamma / m
            i = rv_discrete(values=(actions, p)).rvs()
            opt, reward = self.env.pull(i)
            weights[i] *= np.exp(self.gamma * reward / (p[i] * m))
            choices[t] = i
            opt_rewards[t, ] = np.array([opt, reward])
        return choices, opt_rewards

