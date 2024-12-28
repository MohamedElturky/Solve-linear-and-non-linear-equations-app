import math

from sympy import lambdify, symbols, sympify

class FixedPoint:
    def __init__(self, g, x0, tol=1e-5, max_iter=100, sig=None, step_by_step=False):
        self.g = sympify(g)
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.sig = sig
        self.step_by_step = step_by_step
        self.x = symbols('x')
        self.g = lambdify(self.x, self.g)

    def sign_func(self, value):
        if self.sig is None:
            return value
        if value == 0:
            return 0
        magnitude = math.floor(math.log10(abs(value)))
        scale = 10 ** (self.sig - 1 - magnitude)
        rounded = round(value * scale) / scale
        return rounded

    def apply(self):
        x = self.x0
        iter_count = 0
        ea = 100.0
        steps_string = ""

        if self.step_by_step:
            steps_string += "**** Fixed Point Iteration start ****\n"
            steps_string += f"Initial guess: x0 = {x}\n"

        while ea >= self.tol and iter_count < self.max_iter:
            x_old = x
            try:
                x = self.sign_func(self.g(x_old))
            except (OverflowError, ZeroDivisionError) as e:
                raise ValueError(f"Error during evaluation of g(x): {e}")
            except Exception as e:
                raise ValueError(f"Unexpected error during evaluation of g(x): {e}")

            if x != 0:
                ea = abs((x - x_old) / x)

            iter_count += 1

            if self.step_by_step:
                steps_string += f"Iteration {iter_count}: x = {x}, relative error = {ea}\n"

            if math.isinf(x) or math.isnan(x):
                raise ValueError(f"Iteration diverged at step {iter_count}: x = {x}")

        if ea <= self.tol:
            if self.step_by_step:
                steps_string += f"Converged to {x} after {iter_count} iterations\n"
                steps_string += "**** Fixed Point Iteration end ****\n"
            return x, iter_count, ea, steps_string
        else:
            raise ValueError(f"Iteration did not converge within {self.max_iter} iterations. Last value: {x}")