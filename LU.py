import numpy as np
from GaussElemination import GaussElemination
from Method import Method,Equations
TOL = 1e-7
class LU(Method):
    def __init__(self, coff, sol, method, sig=5, step_by_step=False):
        super().__init__(coff, sol, sig, step_by_step)
        self.method = method

    def apply(self):
        #self.coff = self.sign_array(self.coff)
        if self.method == "Doolittle":
            return self.dooLittle()
        elif self.method == "Crout":
            return self.crout()
        elif self.method == "Cholesky":
            return self.cholesky()
        else:
            raise ValueError("Invalid method")
        

    def dooLittle(self):
          
        n = len(self.coff)
        o = np.zeros(n)
        for i in range (n):
            o[i] =i
        P    = np.identity(n)
        L    = np.identity(n)
        U    = self.coff.copy()
        PF   = np.identity(n)
        LF   = np.zeros((n,n))
        for k in range(0, n - 1):
            index = np.argmax(abs(U[k:, k]))
            index = index + k 
            if index != k:
                P = np.identity(n)
                o[index], o[k] = o[k], o[index]
                P[[index, k], k:n] = P[[k, index], k:n]
                U[[index, k], k:n] = U[[k, index], k:n] 
                PF = np.dot(P, PF)
                LF = np.dot(P, LF)
            L = np.identity(n)
            for j in range(k+1,n):
                L[j, k]  = -(U[j, k] / U[k, k])
                L[j, k]  = self.sign(L[j, k])
                LF[j, k] =  (U[j, k] / U[k, k])
                LF[j, k] =  self.sign(LF[j, k])
            U = np.dot(L,U)
        np.fill_diagonal(LF, 1)
        
        # Solving 
        if self.step_by_step:
            print("Solving using lower and upper :")
            print("Applying gauss elimination then forward sub. on lower to get Y") 

        order = np.array(o, dtype=int)
        sort = self.sol[order]
        outer = GaussElemination(LF,sort,self.sig)
        Y = outer.forwardSub()
        inner = GaussElemination(U,Y,self.sig)
        sol = inner.backSub()

        if self.step_by_step:
            print(f"Solution = {sol}")
            print("**** Doolittle end ****")

        return sol
    
    def crout(self):
        A = self.coff
        tol = TOL

        n = len(A)
        L = np.zeros((n, n))
        U = np.eye(n)
        P = np.eye(n)

        for j in range(n):
            # Pivoting
            max_index = np.argmax(abs(A[j:, j])) + j
            if abs(A[max_index, j]) < tol:
                raise ValueError("Matrix is singular or nearly singular")
            if max_index != j:
                A[[j, max_index]] = A[[max_index, j]]
                P[[j, max_index]] = P[[max_index, j]]

            for i in range(j, n):
                L[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))
            for i in range(j + 1, n):
                U[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(j))) / L[j, j]

            # Solving 
        if self.step_by_step:
            print("Solving using lower and upper :")
            print("Applying gauss elimination then forward sub. on lower to get Y")

        outer = GaussElemination(L,self.sol,self.sig)
        Y = outer.forwardSub()

        if self.step_by_step:
            print(f"Y = {Y}")
            print("Applying gauss elimination then back sub. on upper to get solution")

        inner = GaussElemination(U,Y,self.sig)
        sol = inner.backSub()

        if self.step_by_step:
            print(f"Solution = {sol}")
            print("**** Crout end ****")

        return sol

    # def crout(self):
    #     if self.step_by_step:
    #         print("**** Crout start ****")
    #         print("a = ")
    #         print(self.coff)
    #         print("b = ")
    #         print(self.sol)

    #     if TOL <= 0:
    #         raise ValueError("Tolerance must be a positive number")
        
    #     n = len(self.coff)
    #     lower = np.zeros((n, n))
    #     upper = np.eye(n)
    #     P = np.eye(n)

    #     if self.step_by_step:
    #         print("Lower = ")
    #         print(lower)
    #         print("Upper = ")
    #         print(upper)

    #     for j in range(n):
    #         # Partial Pivoting
    #         max_index = np.argmax(abs(self.coff[j:, j])) + j
    #         if j != max_index:
    #             if self.step_by_step:
    #                 print(f"Applying partial pivoting at row {i}")
    #             self.coff[[j, max_index]] = self.coff[[max_index, j]]
    #             P[[j, max_index]] = P[[max_index, j]]
    #             self.sol[[j, max_index]] = self.sol[[max_index, j]]

    #             if self.step_by_step:
    #                 print("a = ")
    #                 print(self.coff)
    #                 print("b = ")
    #                 print(self.sol)

    #         for i in range(j, n):
    #             sum = 0
    #             for k in range(j):
    #                 sum += self.sign(lower[i, k] * upper[k, j])
    #             lower[i, j] = self.sign(self.coff[i, j] - sum)

    #             if self.step_by_step:
    #                 print(f"lower[{i}][{j}] = {lower[i][j]}")
    #                 print(f"lower = ")
    #                 print(lower)

    #         for i in range(j + 1, n):
    #             sum = 0
    #             for k in range(j):
    #                 sum += self.sign(lower[j, k] * upper[k, i])
                
    #             if self.step_by_step:
    #                 print(f"product = {sum}")

    #             if abs(lower[j, j]) < TOL:
    #                 if self.step_by_step:
    #                     print(f"product is almost 0 which means it's a singular matrix")

    #                 raise ZeroDivisionError("Singular Matrix")
                
    #             upper[j, i] = self.sign((self.coff[j, i] - sum) / lower[j, j])

    #             if self.step_by_step:
    #                 print(f"upper[{j}][{i}] = {upper[j][i]}")
    #                 print(f"upper = ")
    #                 print(upper)

    #     # Solving 
    #     if self.step_by_step:
    #         print("Solving using lower and upper :")
    #         print("Applying gauss elimination then forward sub. on lower to get Y")

    #     outer = GaussElemination(lower,self.sol,self.sig)
    #     Y = outer.forwardSub()

    #     if self.step_by_step:
    #         print(f"Y = {Y}")
    #         print("Applying gauss elimination then back sub. on upper to get solution")

    #     inner = GaussElemination(upper,Y,self.sig)
    #     sol = inner.backSub()

    #     if self.step_by_step:
    #         print(f"Solution = {sol}")
    #         print("**** Crout end ****")

    #     return sol
    
    def cholesky(self):
        if self.step_by_step:
            print("**** Cholesky start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        if TOL <= 0:
            raise ValueError("Tolerance must be a positive number")
        n = len(self.coff)

        if not np.allclose(self.coff, self.coff.T, atol=TOL):
            if self.step_by_step:
                print("Matrix is not symmetric")

            raise ValueError("Input matrix must be symmetric")
        
        lower = np.zeros((n, n))
        P = np.eye(n)

        if self.step_by_step:
            print("Lower = ")
            print(lower)

        for i in range(n):
            # Partial Pivoting
            max_index = np.argmax(abs(self.coff[i:, i])) + i
            if i != max_index:
                if self.step_by_step:
                    print(f"Applying partial pivoting at row {i}")
                self.coff[[i, max_index]] = self.coff[[max_index, i]]
                P[[i, max_index]] = P[[max_index, i]]
                self.sol[[i, max_index]] = self.sol[[max_index, i]]

                if self.step_by_step:
                    print("a = ")
                    print(self.coff)
                    print("b = ")
                    print(self.sol)

            for j in range(i + 1):
                sum = 0
                if j == i:  # Diagonal elements
                    for k in range(j):
                        sum += self.sign(lower[j, k] ** 2)
                    diff = self.sign(self.coff[j, j] - sum)

                    if self.step_by_step:
                        print(f"for [{i}][{j}] , diff = {diff} (diagonal)")

                    if diff < 0:
                        if self.step_by_step:
                            print("As diff < 0 , Matrix is not positive definite")

                        raise ValueError("Matrix is not positive definite")
                    lower[j, j] = self.sign(np.sqrt(diff))

                    if self.step_by_step:
                        print(f"lower[{j}][{j}] = sqrt(diff) = {lower[j][j]}")
                        print(f"lower = ")
                        print(lower)

                else:
                    for k in range(j):
                        sum += self.sign(lower[i, k] * lower[j, k])
                    if abs(lower[j, j]) < TOL:
                        if self.step_by_step:
                            print(f"lower[{j}][{j}] equals (is almost) 0 which means it's a singular matrix")

                        raise ZeroDivisionError("Singular Matrix")
                    lower[i, j] = self.sign((self.coff[i, j] - sum) / lower[j, j])

                    if self.step_by_step:
                        print(f"lower[{i}][{j}] = {lower[i][j]}")
                        print(f"lower = ")
                        print(lower)
                    

        if self.step_by_step:
            print(f"Upper = lower.T = ")
            print(lower.T)
            print()
            print("Solving :")
            print("Applying gauss elimination then forward sub. on lower to get Y")

        # Solving using lower
        outer = GaussElemination(lower,self.sol,self.sig)
        Y = outer.forwardSub()

        if self.step_by_step:
            print(f"Y = {Y}")
            print("Applying gauss elimination then back sub. on upper to get solution")

        inner = GaussElemination(lower.T,Y,self.sig)
        sol = inner.backSub()

        if self.step_by_step:
            print(f"Solution = {sol}")
            print("**** Cholesky end ****")
            
        return sol

if __name__ == "__main__":
    sol = np.array([7, 12, 13])
    #sol = sol.astype(float)
    coff = np.array([[6, 15, 55],[15, 55, 225],[55, 225, 979]])
    #coff = coff.astype(float)
    jr =LU(coff,sol,"Crout")
    print(jr.apply())
    print(np.linalg.solve(coff,sol))

