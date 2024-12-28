import numpy as np
from Method import Method
import math
class GaussSeidel(Method):
    def __init__(self,coff,sol,guess,iter,tol,sig=5, step_by_step=False):
        super().__init__(coff, sol, sig, step_by_step)
        if guess is not None:
            self.guess=np.array(guess, dtype=float)
        else:
            self.guess=np.zeros(len(sol))
        self.iter=iter
        self.tol=tol
        self.n=len(sol)
       

    def apply(self):
        if self.step_by_step:
            print("**** Gauss Seidel start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        if self.guess is not None:
            x=np.array(self.guess,dtype=float)
        else:
            x=np.zeros(self.n)
        
        if self.step_by_step:
            print("Initial guess = ")
            print(x)
        
        if self.iter !=None and self.tol !=None :
            for z in range(self.iter):
                if self.step_by_step:
                    print(f"* Iteration {z+1} *")
                for i in range(self.n):
                    s1=sum(self.coff[i][j]*x[j] for j in range(i))
                    s2=sum(self.coff[i][j]*x[j] for j in range(i + 1, self.n))
                    s1 = self.sign(s1)
                    s2 = self.sign(s2)
                    x[i]=(self.sol[i]-s1-s2)/self.coff[i,i]
                    x[i] =self.sign(x[i])

                    if self.step_by_step:
                        print(f"x{i+1} = {x[i]}")
                if self.step_by_step:
                    print(f"x = {x}")

                if np.linalg.norm(np.dot(self.coff,x)-self.sol,np.inf) < (self.tol*max(1.0,np.linalg.norm(x,np.inf))):
                    if self.step_by_step:
                        print(f"Solution found at iteration {z+1}")
                        print("**** Gauss Seidel end ****")

                    return x , z+1
                
                if self.step_by_step:
                    print(f"didn't converge enough, we have to carry on.")

            if self.step_by_step:
                print(f"Solution not found after {self.iter} iterations")
                print("**** Gauss Seidel end ****")
            return x , self.iter
        
        elif self.iter !=None:
            for z in range(self.iter):
                if self.step_by_step:
                    print(f"* Iteration {z+1} *")

                for i in range(self.n):
                    s1=sum(self.coff[i][j]*x[j] for j in range(i))
                    s2=sum(self.coff[i][j]*x[j] for j in range(i + 1, self.n))
                    s1 = self.sign(s1)
                    s2 = self.sign(s2)
                    x[i]=(self.sol[i]-s1-s2)/self.coff[i,i]
                    x[i] =self.sign(x[i])

                    if self.step_by_step:
                        print(f"x{i+1} = {x[i]}")
                if self.step_by_step:
                    print(f"x = {x}")
            
            if self.step_by_step:
                print("**** Gauss Seidel end ****")

            return x ,self.iter
        else:
            iteration = 0
            while True:
                if self.step_by_step:
                    print(f"* Iteration {iteration+1} *")

                iteration += 1
                for i in range(self.n):
                    s1=sum(self.coff[i][j]*x[j] for j in range(i))
                    s2=sum(self.coff[i][j]*x[j] for j in range(i + 1, self.n))
                    s1 = self.sign(s1)
                    s2 = self.sign(s2)
                    x[i]=(self.sol[i]-s1-s2)/self.coff[i,i]
                    x[i] =self.sign(x[i])

                    if self.step_by_step:
                        print(f"x{i+1} = {x[i]}")
                if self.step_by_step:
                    print(f"x = {x}")

                if np.linalg.norm(np.dot(self.coff,x)-self.sol,np.inf) < (self.tol*max(1.0,np.linalg.norm(x,np.inf))):
                    return x , iteration
                if iteration > 100:
                    break

                if self.step_by_step:
                    print(f"didn't converge enough, we have to carry on.")

            return x , iteration

