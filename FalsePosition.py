import math
import time
from sympy import symbols, sympify, lambdify

class FalsePosition:
    def __init__(self, equation, lower, upper, epsilon=1e-5, max_iterations=50, sig=None, step_by_step=False):
        self.equation = sympify(equation)
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.sig = sig
        self.step_by_step = step_by_step
        self.x = symbols('x')
        self.function = lambdify(self.x, self.equation)

    def sign_func(self, value):
        if self.sig is None:
            return value
        if value == 0:
            return 0
        magnitude = math.floor(math.log10(abs(value)))
        scale = 10 ** (self.sig - 1 - magnitude)
        rounded = round(value * scale) / scale
        return rounded

    def solve(self):
        a, b = self.lower, self.upper
        steps_string = ""
        if self.function(a) * self.function(b) > 0:
            raise ValueError("No root found. The function must have opposite signs at the interval boundaries.")
        elif self.function(a) == 0:
            return a, 0, 0, steps_string
        elif self.function(b) == 0:
            return b, 0, 0, steps_string
        c_old = 0
        for iteration in range(1, self.max_iterations + 1):
            fa = self.function(a)
            fb = self.function(b)
            c = self.sign_func((a * fb - b * fa) / (fb - fa))
            fc = self.function(c)
            if self.step_by_step:
                steps_string += f"Iteration {iteration}: a = {a}, b = {b}, c = {c}, f(c) = {fc}\n"

            if abs(fc) == 0:
                if self.step_by_step:
                    steps_string += f"Converged to {c} after {iteration} iterations\n"
                return c, iteration, 0, steps_string
            if abs((c - c_old) / c) < self.epsilon and iteration > 1:
                if self.step_by_step:
                    steps_string += f"Converged to {c} after {iteration} iterations\n"
                return c, iteration, abs((c - c_old) / c) , steps_string
            c_old = c
            if self.function(a) * fc < 0:
                b = c
            else:
                a = c

        raise ValueError(f"Iteration did not converge within {self.max_iterations} iterations. Last interval: [{a}, {b}]")