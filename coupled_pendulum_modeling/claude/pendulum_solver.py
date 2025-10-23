"""
Pendulum Simulation Solver
Solves the coupled pendulum system and saves results to file
"""

import numpy as np
from scipy.integrate import solve_ivp
import pickle


class PendulumSimulator:
    def __init__(self):
        # Physical parameters
        self.c1 = 3.22e-9
        self.c2 = 1 * self.c1
        m = 1.4e-5
        self.m_pen = m / 4
        d = 2e-3
        self.J = 1/6 * m * d**2
        self.B = 1e-6
        self.wb = 40 * 2 * np.pi
        self.g = 9.81
        self.l = 1e-3  # Pendulum length
        self.B_inf = 1e-7 * 0
        self.Ff = 1e-7
        self.x_range = 4 * np.pi
        self.x_offset = np.pi / 4
        
        # Time span and initial conditions
        self.tspan = [0, 0.4]
        self.y0 = [0, 0, 0, 0]
        
    def model(self, t, x):
        """
        Pendulum dynamics model
        State vector: x = [theta1, theta1_dot, theta2, theta2_dot]
        """
        x1, x2, x3, x4 = x
        
        # Resonance force on pendulum 1
        F_res1 = -self.Ff * np.exp(-((x1 - self.x_offset) * self.x_range)**2) * x2
        F_res2 = 0
        
        # State derivatives
        x1d = x2
        x3d = x4
        
        # Angular accelerations
        x2d = (self.B * np.sin(self.wrap_to_pi(self.wb * t - x1)) + 
               self.B_inf * np.sin(self.wrap_to_pi(x3 - x1)) - 
               self.c1 * x2 - 
               self.m_pen * self.g * self.l * np.cos(np.pi - x1) + 
               F_res1) / self.J
        
        x4d = (self.B * np.sin(self.wrap_to_pi(self.wb * t - x3)) + 
               self.B_inf * np.sin(self.wrap_to_pi(x1 - x3)) - 
               self.c2 * x4 - 
               self.m_pen * self.g * self.l * np.cos(np.pi - x3) + 
               F_res2) / self.J
        
        return [x1d, x2d, x3d, x4d]
    
    @staticmethod
    def wrap_to_pi(angle):
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def wrap_to_2pi(angle):
        """Wrap angle to [0, 2*pi]"""
        return angle % (2 * np.pi)
    
    def run_simulation(self):
        """Run the ODE solver"""
        print('Running simulation...')
        
        # Use solve_ivp (similar to MATLAB's ode15s)
        sol = solve_ivp(
            self.model,
            self.tspan,
            self.y0,
            method='BDF',  # Similar to ode15s (stiff solver)
            dense_output=True,
            rtol=1e-6,
            atol=1e-9
        )
        
        print(f'Simulation finished. Time points: {len(sol.t)}')
        
        return sol.t, sol.y.T  # Return t and y with shape (n_timepoints, 4)
    
    def save_results(self, filename='pendulum_data.pkl'):
        """Run simulation and save results to file"""
        t, y = self.run_simulation()
        
        # Package data
        data = {
            't': t,
            'y': y,
            'params': {
                'c1': self.c1,
                'c2': self.c2,
                'm_pen': self.m_pen,
                'J': self.J,
                'B': self.B,
                'wb': self.wb,
                'g': self.g,
                'l': self.l,
                'B_inf': self.B_inf,
                'Ff': self.Ff,
                'x_range': self.x_range,
                'x_offset': self.x_offset
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f'Results saved to {filename}')
        return t, y


if __name__ == '__main__':
    # Create simulator and run
    sim = PendulumSimulator()
    t, y = sim.save_results()
    
    print(f'\nSimulation complete!')
    print(f'Final time: {t[-1]:.4f} s')
    print(f'Data shape: {y.shape}')