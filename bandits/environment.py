import numpy as np

class Environment(object):    
    
    def __init__(self, arms, ctx_gen=None):
        self.arms = arms
        self.ctx_gen = ctx_gen 
        self.ctx = self._generate_context() 
        
    def m(self):
        return len(self.arms)
        
    def pull(self, i):
        r = self.arms[i].pull(self.ctx)
        self.ctx = self._generate_context() 
        return max(a.expected(self.ctx) for a in self.arms), r

    def get_context(self):
        return self.ctx

    def _generate_context(self):
        return None if self.ctx_gen is None else self.ctx_gen.generate_context()
        
class ContextGenerator(object):

    def get_context(self):
        raise NotImplementedError


class Arm(object):
    
    def pull(self, *ctx):
        raise NotImplementedError

    def expected(self):
        raise NotImplementedError

class BaseArm(Arm):
    
    def __init__(self, D):
        self.D = D
    
    def pull(self, ctx):
        return self.D.rvs()
    
    def expected(self, ctx):
        return self.D.mean()

