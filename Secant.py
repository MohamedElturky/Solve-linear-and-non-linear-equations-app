import math
from sympy import lambdify, symbols, sympify

class Secant:
    def __init__(self, f, x0, x1, tol=1e-5, max_iter=100, sig=None, step_by_step=False):
        self.f = sympify(f)
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.max_iter = max_iter
        self.sig = sig
        self.step_by_step = step_by_step
        self.x = symbols('x')
        self.f = lambdify(self.x, self.f)

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
        x0, x1 = self.x0, self.x1
        iter_count = 0
        ea = 100.0
        steps_string = ""

        if self.step_by_step:
            steps_string += "**** Secant Method start ****\n"
            steps_string += f"Initial guesses: x0 = {x0}, x1 = {x1}\n"

        if abs(x1 - x0) < self.tol:
            if self.step_by_step:
                steps_string += f"Converged to {x1} after {iter_count} iterations\n"
                steps_string += "**** Secant Method end ****\n"
            return x1, iter_count, ea, steps_string

        while ea > self.tol and iter_count < self.max_iter:
            try:
                f_x0 = self.f(x0)
                f_x1 = self.f(x1)
                if f_x1 - f_x0 == 0:
                    raise ZeroDivisionError("Division by zero in Secant method")
                x2 = self.sign_func(x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0))
            except (OverflowError, ZeroDivisionError) as e:
                raise ValueError(f"Error during evaluation of f(x): {e}")
            except Exception as e:
                raise ValueError(f"Unexpected error during evaluation of f(x): {e}")

            if x2 != 0:
                ea = abs((x2 - x1) / x2)

            iter_count += 1

            if self.step_by_step:
                steps_string += f"Iteration {iter_count}: x2 = {x2}, relative error = {ea}\n"

            if math.isinf(x2) or math.isnan(x2):
                raise ValueError(f"Iteration diverged at step {iter_count}: x2 = {x2}")

            x0, x1 = x1, x2

            if ea < self.tol:
                if self.step_by_step:
                    steps_string += f"Converged to {x2} after {iter_count} iterations\n"
                    steps_string += "**** Secant Method end ****\n"
                return x2, iter_count, ea, steps_string

        raise ValueError(f"Iteration did not converge within {self.max_iter} iterations. Last value: {x2}")
