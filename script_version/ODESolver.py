import numpy as np

"""
Script to solve a general ODE, stepped as per the Forward Euler Method
local error is proportional to square(step_size) --> first order Differential Equation
global error is proportional to (step size)
"""

class ODESolver:
    """
    Superclass for solving ODEs (using Euler method - subclass).
    """
    def __init__(self, model):
        self.model = model

    def step(self):
        """
        Advance solution one time step. Implemented in subclass.
        """
        pass

    def set_init_conditions(self, U0):
        #Scalar ODE
        if isinstance(U0, (int, float)):
            self.n_of_eqns = 1
            U0 = float(U0)
        else: #System of eqns
            U0 = np.asarray(U0)
            self.n_of_eqns = U0.size
        self.U0 = U0

    def solve(self, time_points):
        self.t = np.asarray(time_points)
        n = self.t.size #no of points = size of time steps array (no. of measurements) ---> npts = ndays/dt
        self.u = np.zeros((n, self.n_of_eqns))
        self.u[0, :] = self.U0 #First timepoint of the entire solution array (all 3 equations) set to the initial conditions
        # Integrate
        for i in range(n-1):
            self.i = i
            self.u[i + 1] = self.step() #Computing from second row; first row was initial conditions
        return self.u[:i+2], self.t[:i+2] #loop is till n-1 and range is outer limit exclusive


class EulerMethod(ODESolver):
    """
    u(t+1) = u(t) + (model * stepsize(x))
    """
    def step(self): #Inheriting from superclass - override method
        u, model, i, t = self.u, self.model, self.i, self.t
        dt = t[i + 1] - t[i] #stepsize
        return u[i, :] + dt * model(u[i, :], t[i])
