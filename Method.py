import numpy as np
import math
class Equations():

    def __init__(self,num):
        self.i=0
        self.j=0
        self.sig= 5
        self.coff = np.zeros((num,num))
        self.sol = np.zeros(num)
        self.num =num
    def setCoff(self,b):
        if(self.j == self.num):
            self.sol[self.i] = b
            self.i +=1
            self.j = 0
        else:
            self.coff[self.i,self.j] = b
            self.j += 1

class Method():
    def __init__(self,coff,solu,sig=5 , step_by_step = False):
        # self.coff = coff
        self.coff = np.array(coff, dtype=float)
        self.sol =  np.array(solu , dtype=float)
        self.sig = sig
        self.step_by_step = step_by_step

    def __str__(self):
        return f"{self.coff} = {self.sol}"
    
    def pivoting(self,i,j):
        if self.step_by_step:
            print(f"**** Pivoting on a{i}{j} start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        row = i
        max = self.coff[i,j]
        for k in range(i,len(self.coff)):
            if (abs(self.coff[k,j]) > max):
                max =self.coff[k,j]
                row = k
        self.coff[[i, row], :] = self.coff[[row, i], :]

        self.sol[i],self.sol[row] = self.sol[row],self.sol[i]

        if self.step_by_step:
            print(f"After pivoting on a{i}{j}")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)
            print(f"**** Pivoting on a{i}{j} end ****")
    
    
    def forwardElemination(self,i,j):
        if self.step_by_step:
            print(f"**** Forward elemination on a{i}{j} start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)
        
        self.pivoting(i,j)
        pivot = self.coff[i, i]
        if pivot == 0 and self.sol[i] == 0:
            if self.step_by_step:
                print(f"Pivot[{i}] and  b[{i}] = 0 , Infinite Number of Solutions")
                print(f"**** Forward elemination on a{i}{j} end ****")

            raise ValueError("Infinite Number of Solutions")
        if pivot == 0:
            if self.step_by_step:
                print(f"Pivot[{i}] = 0 while b[{i}] !=0 , No Solution (Singular matrix)")
                print(f"**** Forward elemination on a{i}{j} end ****")

            raise ValueError("No Solution (Singular matrix)")
        
        for k in range (i+1,len(self.coff)):
            m = self.sign(self.coff[k,j]/self.coff[i,j])
            self.coff[k,j] =0
            self.sol[k] -= m*self.sol[i]
            self.sol[k]= self.sign(self.sol[k])

            if self.step_by_step:
                print(f"m = {m}")
                print(f"b{k} = b{k} - m*b{i} = {self.sol[k]}")
                

            for l in range(j+1,len(self.coff)):
                self.coff[k,l] -= m*self.coff[i,l]
                self.coff[k,l]= self.sign(self.coff[k,l])
                
                if self.step_by_step:
                    print(f"a{k}{l} = a{k}{l} - m*a{i}{l} = {self.coff[k,l]}")
                    
            
        if self.step_by_step:
            print(f"after forward elemination on a{i}{j}")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        if(i == len(self.coff)-1 and j == len(self.coff)-1):
            if self.step_by_step:
                print(f"**** Forward elemination on a{i}{j} end ****")
                
            return True
        else:
            if self.step_by_step:
                print(f"**** Forward elemination on a{i}{j} end ****")      

            return self.forwardElemination(i+1,j+1)

    def backSub(self):

        if self.step_by_step:
            print("**** Backward Substitution start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        x = np.zeros(len(self.sol))

        for k in range(len(self.coff) - 1, -1, -1):

            sum = self.sol[k]
        

            for l in range(k + 1, len(self.coff)):
                sum -= self.coff[k, l] * x[l]
        

            x[k] =self.sign( sum / self.coff[k, k])

            if self.step_by_step:
                print(f"X{k} = {x[k]}")

        if self.step_by_step:
            print(f"Solution = {x}")
            print("**** Backward Substitution end ****")

        return x
    
    def forwardSub(self):
        if self.step_by_step:
            print("**** Forward Substitution start ****")
            print("a = ")
            print(self.coff)
            print("b = ")
            print(self.sol)

        x = np.zeros(len(self.sol))
        for k in range(len(self.coff)):
            sum = self.sol[k]
            for l in range(k):
                sum -= self.coff[k, l] * x[l]
        
            x[k] =self.sign( sum / self.coff[k, k])

            if self.step_by_step:
                print(f"X{k} = {x[k]}")

        return x

    def sign (self,value):
        if value == 0:
            return 0
        magnitude = math.floor(math.log10(abs(value)))
        scale =  10 ** (self.sig -1 - magnitude)
        rounded = round(value * scale) / scale
        return rounded
    
    def sign_array(self, array):
        return np.array([self.sign(val) for val in array])