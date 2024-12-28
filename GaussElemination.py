import numpy as np
from Method import Method,Equations

class GaussElemination(Method):
    def __init__(self,coff,sol,sig=5 , step_by_step=False):
        super().__init__(coff,sol,sig, step_by_step)
    def apply(self):
        return self.backSub()


