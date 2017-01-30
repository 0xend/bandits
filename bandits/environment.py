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

