import numpy as np
from Method import Method, Equations

import numpy as np

class GaussJordan(Method):
    def __init__(self, coff, sol, sig=5, step_by_step=False):
        super().__init__(coff, sol, sig, step_by_step)

    def apply(self):
        if self.step_by_step:
            print("**** Gauss Jordan start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        #self.forwardElimination()
        self.reducedEchelon()

        if self.step_by_step:
            print (f"solution = {self.sol}")
            print("**** Gauss Jordan end ****")

        return self.sol  # Round final results for presentation

    def forwardElimination(self):
        n = len(self.coff)
        for i in range(n):
            self.pivoting(i, i)  # Ensure pivoting for numerical stability
            pivot = self.coff[i, i]
            if pivot == 0 and self.sol[i] == 0:
                raise ValueError("Infinite Number of Solutions")
            if pivot == 0:
                raise ValueError("No Solution (Singular matrix)")

            # Normalize pivot row
            self.coff[i] = self.sign_array(self.coff[i] / pivot)  # Normalize and round
            self.sol[i] = self.sign(self.sol[i] / pivot)          # Round solution

            # Eliminate all rows below
            for j in range(i + 1, n):
                factor = self.coff[j, i]
                self.coff[j] = self.sign_array(self.coff[j] - factor * self.coff[i])  # Eliminate and round
                self.sol[j] = self.sign(self.sol[j] - factor * self.sol[i])           # Round solution

    def reducedEchelon(self):
        if self.step_by_step:
            print("**** Reduced Echelon Form start ****")

        n = len(self.coff)
        for i in range(n - 1, -1, -1):  # Start from the last row
            # Normalize the pivot row
            pivot = self.coff[i, i]
            if pivot != 0:
                self.coff[i] = self.sign_array(self.coff[i] / pivot)  # Normalize and round
                self.sol[i] = self.sign(self.sol[i] / pivot)          # Round solution

                if self.step_by_step:
                    print(f"Normalize row {i} by dividing it by {pivot}:")
                    print(f"a = ")
                    print(self.coff)
                    print(f"b = ")
                    print(self.sol)

            if self.step_by_step:
                print(f"Eliminate all elements above row {i}")

            # Eliminate all rows above
            for j in range(i - 1, -1, -1):
                factor = self.coff[j, i]
                self.coff[j] = self.sign_array(self.coff[j] - factor * self.coff[i])  # Eliminate and round
                self.sol[j] = self.sign(self.sol[j] - factor * self.sol[i])           # Round solution

            if self.step_by_step:
                # print(f"Row {i} is done")
                print("a = ")
                print(self.coff)
                print("b = ")
                print(self.sol)
        if self.step_by_step:
            print("**** Reduced Echelon Form end ****")

