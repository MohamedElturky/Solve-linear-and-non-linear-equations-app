import numpy as np
from Method import Method
import math

class Jacobi(Method):
    def __init__(self, coff, sol, guess, iter, tol, sig=5, step_by_step=False):
        super().__init__(coff, sol, sig, step_by_step)
        if guess is not None:
            self.guess = np.array(guess, dtype=float)
        else:
            self.guess = np.zeros(len(sol))
        self.iter = iter
        self.tol = tol
        self.n = len(sol)

    def apply(self):
        if self.step_by_step:
            print("**** Jacobi start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        for i in range(self.n):
            self.pivoting(i, i)

        if self.guess is not None:
            x = np.array(self.guess, dtype=float)
        else:
            x = np.zeros(self.n)

        if self.step_by_step:
            print("Initial guess = ")
            print(x)

        if self.iter is not None and self.tol is not None:
            for z in range(self.iter):
                if self.step_by_step:
                    print(f"* Iteration {z+1} *")

                y = np.zeros_like(x)
                for i in range(self.n):
                    s = sum(self.coff[i][j] * x[j] for j in range(self.n) if j != i)
                    s = self.sign(s)
                    y[i] = (self.sol[i] - s) / self.coff[i, i]
                    y[i] = self.sign(y[i])

                if self.step_by_step:
                    print(f"y = {y}")

                if np.linalg.norm(y - x, np.inf) < (self.tol * max(1.0, np.linalg.norm(y, np.inf))):
                    if self.step_by_step:
                        print(f"Solution found at iteration {z+1}")
                        print("**** Jacobi end ****")

                    return y, z+1

                if self.step_by_step:
                    print(f"didn't converge enough, we have to carry on.")

                x = y
            return y, self.iter
        elif self.iter is not None:
            for z in range(self.iter):
                if self.step_by_step:
                    print(f"* Iteration {z+1} *")

                y = np.zeros_like(x)
                for i in range(self.n):
                    s = sum(self.coff[i][j] * x[j] for j in range(self.n) if j != i)
                    s = self.sign(s)
                    y[i] = (self.sol[i] - s) / self.coff[i, i]
                    y[i] = self.sign(y[i])
                x = y
            return y, self.iter
        else:
            y = np.zeros_like(x)
            iteration = 0
            while True:
                if self.step_by_step:
                    print(f"* Iteration {iteration+1} *")

                iteration += 1
                y = np.zeros_like(x)
                for i in range(self.n):
                    s = sum(self.coff[i][j] * x[j] for j in range(self.n) if j != i)
                    s = self.sign(s)
                    y[i] = (self.sol[i] - s) / self.coff[i, i]
                    y[i] = self.sign(y[i])

                if np.linalg.norm(y - x, np.inf) < (self.tol * max(1.0, np.linalg.norm(y, np.inf))):
                    if self.step_by_step:
                        print(f"Solution found at iteration {iteration}")
                        print("**** Jacobi end ****")

                    break
                if iteration > 100:
                    break
                x = y
            return y, iteration
