# simplex optimization

import numpy as np
import matplotlib.pyplot as plt

def cost(x):
    x = x + np.array([3, 3])
    return np.sum(x**2, axis=-1)


class SimplexOptimization:

    def __init__(self, cost):
        self.cost = cost
        self.simplex = None
        self.best = None
        self.centroid = None
        self.f = None
        self.iterates = []

    def initialize_simplex(self, dim=2, initial_guess=None, step=2.0):
        """
        Initialize a simplex around an initial guess.
        If no initial guess is provided, use the origin.
        """
        if initial_guess is None:
            initial_guess = np.zeros(dim)
        self.simplex = np.stack([initial_guess for _ in range(dim+1)])
        self.simplex[1:] += np.eye(dim) * step
        
        self.iterates = [self.simplex.copy()]

    def update_centroid(self):
        """assuming the simplex is sorted, the centroid is the average of the best n - 1 points"""
        self.centroid = np.mean(self.simplex[:-1], axis=0)

    def evaluate(self, x):
        """evaluate the cost of a point"""
        return self.cost(x)

    @staticmethod
    def move(x1: np.ndarray, x2: np.ndarray, scale: float):
        """
        all movements can be expressed as a linear combination of 2 points, 
        this serves as the basis for all movements
        """
        return x1 + scale * (x1 - x2)

    def reflect(self, x, alpha):
        """reflect x about the centroid of the other n-1 points"""
        return self.centroid - alpha * (x - self.centroid)

    def expand(self, x, gamma):
        """expand x away from the centroid"""
        return self.centroid + gamma * (x - self.centroid)

    def contract(self, x, rho):
        """contract x towards the centroid"""
        return self.centroid + rho * (x - self.centroid)
    
    def shrink(self, x, sigma):
        """shrink x towards the best point"""
        return self.best + sigma * (x - self.best)

    def sort(self):
        """sort the simplex by cost"""
        idx = np.argsort(self.f)
        self.simplex = self.simplex[idx]
        self.f = self.f[idx]
        self.best = self.simplex[0]

    def param_checks(self, dim, alpha, beta, gamma, rho, sigma, tol, max_iter):
        # Parameter validation
        if not (0 < alpha < 2):
            raise ValueError("Reflection coefficient alpha must be between 0 and 2.")
        if not (0 < beta < 1):
            raise ValueError("Contraction coefficient beta must be between 0 and 1.")
        if gamma < 1:
            raise ValueError("Expansion coefficient gamma must be greater than or equal to 1.")
        if not (0 < rho < 1):
            raise ValueError("Contraction coefficient rho must be between 0 and 1.")
        if not (0 < sigma < 1):
            raise ValueError("Shrink coefficient sigma must be between 0 and 1.")


    def check_terminal_conditions(self, tol, max_iter):
        simplex_size = np.max(self.simplex, axis=0) - np.min(self.simplex, axis=0)
        if np.all(simplex_size < tol):
            print("Terminated due to simplex size below tolerance.")
            return True
        
        if np.abs(self.f[0] - self.f[-1]) < tol:
            print("Terminated due to function value convergence.")
            return True


    def optimize(self, dim=2, alpha=1, beta=0.5, gamma=2, rho=0.5, sigma=0.5, tol=1e-6, max_iter=1000, log_every=100):
        self.param_checks(dim, alpha, beta, gamma, rho, sigma, tol, max_iter)
        self.initialize_simplex(dim)

        self.f = np.array([self.evaluate(xi) for xi in self.simplex])
        self.sort()
        self.update_centroid()

        for i in range(max_iter):
            # reflect
            xr = self.reflect(self.simplex[-1], alpha)
            fr = self.evaluate(xr)
            # if xr is better than the second worst point but not better than the best point
            # replace the worst point with xr
            if self.f[0] <= fr < self.f[-2]:
                self.simplex[-1] = xr
                self.f[-1] = fr
            # if xr is better than the best point
            # keep moving in that direction
            elif fr < self.f[0]:
                # expand
                xe = self.expand(xr, gamma)
                fe = self.evaluate(xe)
                if fe < fr:
                    self.simplex[-1] = xe
                    self.f[-1] = fe
                else:
                    self.simplex[-1] = xr
                    self.f[-1] = fr
            else:
                if fr < self.f[-1]:
                    self.simplex[-1] = xr
                    self.f[-1] = fr
                # contract
                xc = self.contract(self.simplex[-1], rho)
                fc = self.evaluate(xc)
                if fc < self.f[-1]:
                    self.simplex[-1] = xc
                    self.f[-1] = fc
                else:
                    # shrink
                    for i in range(1, dim+1):
                        self.simplex[i] = self.shrink(self.simplex[i], sigma)
                        self.f[i] = self.evaluate(self.simplex[i])
            self.sort()
            self.update_centroid()
            self.iterates.append(self.simplex)
            if self.check_terminal_conditions(tol, max_iter):
                break

            if i % log_every == 0:
                print(f"Iteration {i}: Best cost = {self.f[0]}, Worst cost = {self.f[-1]}")

    def visualize(self):
        if self.simplex.shape[1] != 2:
            raise NotImplementedError("Visualization is only supported for 2D problems.")
        
        x = np.linspace(-5, 5, 400)
        y = np.linspace(-5, 5, 400)
        X, Y = np.meshgrid(x, y)
        Z = self.cost(np.stack([X, Y], axis=-1))
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        
        for simplex in self.iterates:
            plt.plot(simplex[:, 0], simplex[:, 1], 'r-', alpha=0.5)
            plt.plot(simplex[:, 0], simplex[:, 1], 'bo', markersize=3)
        
        plt.title('Simplex Optimization Path')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()


simplex = SimplexOptimization(cost)
simplex.optimize()
simplex.visualize()

