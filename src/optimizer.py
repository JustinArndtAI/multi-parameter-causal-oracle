import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
from src.simulator import Simulator

class BayesianOptimizer:
    """
    Handles the optimization process by wrapping the robust skopt.gp_minimize function.
    This version reverts to a simpler, more stable design while keeping the
    aggressively exploratory settings.
    """
    def __init__(self, ground_truth_trajectory, search_space, simulation_steps, impulses):
        self.ground_truth_trajectory = ground_truth_trajectory
        self.search_space = search_space
        self.simulation_steps = simulation_steps
        self.impulses = impulses
        self.objective_call_count = 0

    def _calculate_rmse(self, trajectory1, trajectory2):
        """Calculates the Root Mean Squared Error between two trajectories."""
        t1 = np.array(trajectory1)
        t2 = np.array(trajectory2)
        
        len1, len2 = len(t1), len(t2)
        if len1 > len2:
            padding = np.tile(t2[-1], (len1 - len2, 1))
            t2 = np.vstack([t2, padding])
        elif len2 > len1:
            padding = np.tile(t1[-1], (len2 - len1, 1))
            t1 = np.vstack([t1, padding])

        return np.sqrt(np.mean((t1 - t2)**2))

    def _objective(self, params):
        """
        The objective function to minimize. It runs a simulation with the given
        parameters and returns the RMSE compared to the ground truth.
        """
        self.objective_call_count += 1
        
        sim_params = {
            'friction': params[0],
            'elasticity': params[1],
            'mass': params[2]
        }
        
        simulator = Simulator(params=sim_params)
        sim_trajectory = simulator.run_simulation_for_trajectory(
            steps=self.simulation_steps,
            impulses=self.impulses
        )
        
        rmse = self._calculate_rmse(self.ground_truth_trajectory, sim_trajectory)
        
        print(f"  Guess #{self.objective_call_count}: (f={params[0]:.4f}, e={params[1]:.4f}, m={params[2]:.4f}) -> RMSE: {rmse:.2f}")
        return rmse

    def run_optimization(self, n_calls):
        """
        Runs the full optimization process.
        """
        print("--- V2 - Starting Robust Bayesian Optimization ---")
        
        result = gp_minimize(
            func=self._objective,
            dimensions=self.search_space,
            n_calls=n_calls,
            # --- Critical Exploration Settings ---
            n_initial_points=int(n_calls / 2), # Radical exploration
            acq_func="LCB",                    # Use Lower Confidence Bound
            kappa=10,                          # Prioritize exploration
            # -------------------------------------
            random_state=42,
            verbose=False
        )

        return result

