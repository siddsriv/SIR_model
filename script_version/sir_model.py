"""
SIR epidemic spread model

S: Susceptible;     I: Infectives;        R:Recovered/Dead = Removed
S + I + R = N (total population)
Transitions only between S -> I -> R

Differential Equation form (wrt time)
dS = -beta * S * I
dI = beta * S * I - gamma * I
dR = gamma * I

Transition rates
beta: Contact rate (coming in contact and getting infected)
gamma: mean recovery rate, 1/gamma is the mean period of time during which an infected individual can pass it on
"""

import numpy as np
from ODESolver import EulerMethod
from matplotlib import pyplot as plt

class SIR:
    def __init__(self, gamma, beta, S0, I0, R0):
        """
        gamma, beta: parameters in the ODE system of eqns
        S0, I0, R0: initial values
        """
        self.S0 = S0
        self.I0 = I0
        #total_people = S0 + I0 + R0
        if isinstance(gamma, (float, int)): #Is it a number?
            self.gamma = lambda t: gamma
        elif callable(gamma): #Make transition rates amenable to change as time progresses
            self.gamma = gamma
        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta
        elif callable(beta):
            self.beta = beta
        self.initial_conditions = [S0, I0, R0]

    def __str__(self):
        return 'Intial susceptible people: {} \nInitial Infected people: {}'.format(self.S0, self.I0)

    def __call__(self, u, t):
        S, I, _ = u
        #Differential Equations
        return np.asarray([
            -self.beta(t)*S*I,                      #Susceptibles
            self.beta(t)*S*I - self.gamma(t)*I,     #Infectives
            self.gamma(t)*I                         #Recovered
        ])



if __name__ == "__main__":
    beta = lambda t: 0.0005 if t <=10 else 0.0001 #SOCIAL DISTANCING/PRECAUTIONS ETC. after first 10 days
    sir = SIR(0.1, beta, 1400, 1, 0) #Initial conditions of the epidemic
    solver = EulerMethod(sir)
    solver.set_init_conditions(sir.initial_conditions)
    #Decide time steps
    ndays = 90
    resolution = 1000
    time_steps, retstep = np.linspace(0, ndays, resolution, retstep=True) #90 days; 1000 measurements(time steps or delta(x) for the Euler method)
    print(sir)
    print("Delta (timestep in next measurement): {}".format(retstep)) #Debugging time steps
    #Solve differential equation
    u, t = solver.solve(time_steps)
    #Plot SIR
    print("\nPlotting for {} days with a resolution of {} \n".format(ndays, resolution))
    plt.xkcd()
    plt.plot(t, u[:, 0], label="Susceptible")
    plt.plot(t, u[:, 1], label="Infected")
    plt.plot(t, u[:, 2], label="Recovered")
    plt.xlabel("Days")
    plt.ylabel("People")
    plt.legend()
    plt.show()
