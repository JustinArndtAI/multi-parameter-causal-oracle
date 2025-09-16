import math
from src.simulator import Simulator
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os

# --- V2 Configuration ---
GROUND_TRUTH_PARAMS = {
    'friction': 0.7,
    'elasticity': 0.9,
    'mass': 12.0
}

# --- Definitive Staged Experiment Design ---

# Stage A: A "bounce-only" experiment to isolate Elasticity and Mass.
EXP_A_IMPULSES = [(0, (0, 8000))]
EXP_A_STEPS = 300

# Stage B: A "slide-only" experiment to isolate Friction.
EXP_B_IMPULSES = [(0, (30000, 0))]
EXP_B_STEPS = 300

# Stage C: The combined experiment for final refinement and plotting.
EXP_C_IMPULSES = [(0, (8000, 8000)), (300, (25000, 0))]
EXP_C_STEPS = 500


# --- Global state for tracking ---
call_count = 0
current_stage = ""

def calculate_rmse(t1, t2):
    """Calculates the Root Mean Squared Error between two trajectories."""
    t1, t2 = np.array(t1), np.array(t2)
    return np.sqrt(np.mean((t1 - t2)**2))

def objective_function(params, search_space_names, fixed_params, ground_truth_traj, steps, impulses):
    """General objective function for our staged optimization."""
    global call_count
    call_count += 1
    
    # Combine the parameters being searched with the fixed ones
    current_params = fixed_params.copy()
    for i, name in enumerate(search_space_names):
        current_params[name] = params[i]
        
    simulator = Simulator(params=current_params)
    sim_trajectory = simulator.run_simulation_for_trajectory(steps=steps, impulses=impulses)
    
    rmse = calculate_rmse(ground_truth_traj, sim_trajectory)
    
    param_str = ", ".join([f"{k}={v:.2f}" for k, v in current_params.items()])
    print(f"  {current_stage} Guess #{call_count}: ({param_str}) -> RMSE: {rmse:.2f}")
    return rmse

def plot_results(ground_truth_traj, initial_guess_params, final_params, impulses, steps, filename):
    """
    Generates and saves a plot comparing the ground truth, initial guess,
    and final calibrated trajectories.
    """
    print(f"\n--- Generating final plot... ---")
    
    # Get the trajectory for the initial incorrect guess
    initial_sim = Simulator(params=initial_guess_params)
    initial_traj = initial_sim.run_simulation_for_trajectory(steps=steps, impulses=impulses)

    # Get the trajectory for the final calibrated parameters
    final_sim = Simulator(params=final_params)
    final_traj = final_sim.run_simulation_for_trajectory(steps=steps, impulses=impulses)

    # Unzip trajectories for plotting
    gt_x, gt_y = zip(*ground_truth_traj)
    initial_x, initial_y = zip(*initial_traj)
    final_x, final_y = zip(*final_traj)

    plt.figure(figsize=(12, 8))
    plt.plot(gt_x, gt_y, 'g-', label='Ground Truth Trajectory', linewidth=4, alpha=0.8)
    plt.plot(initial_x, initial_y, 'r--', label='Initial Guess Trajectory', linewidth=2)
    plt.plot(final_x, final_y, 'b:', label='Calibrated Trajectory', linewidth=2, alpha=0.9)
    
    plt.title('Causal Oracle Calibration Results', fontsize=16)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Ensure the figures directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    print(f"Plot saved successfully to {filename}")


def run_staged_calibration():
    """
    Runs the entire V2 experiment using the definitive staged approach.
    """
    initial_guess = {'friction': 0.5, 'elasticity': 0.5, 'mass': 15.0} # Start with a neutral guess
    best_params = initial_guess.copy()
    global call_count, current_stage

    # === STAGE A: OPTIMIZE ELASTICITY & MASS ===
    current_stage = "Stage A (Bounce)"
    call_count = 0
    print(f"\n--- {current_stage}: Isolating Elasticity and Mass ---")
    
    sim_a = Simulator(GROUND_TRUTH_PARAMS)
    traj_a = sim_a.run_simulation_for_trajectory(steps=EXP_A_STEPS, impulses=EXP_A_IMPULSES)
    
    search_space_a = [Real(0.1, 1.0, name='elasticity'), Real(5.0, 25.0, name='mass')]
    
    obj_a = partial(objective_function, 
                    search_space_names=['elasticity', 'mass'],
                    fixed_params={'friction': best_params['friction']},
                    ground_truth_traj=traj_a, steps=EXP_A_STEPS, impulses=EXP_A_IMPULSES)

    result_a = gp_minimize(func=obj_a, dimensions=search_space_a, n_calls=50, n_initial_points=25, random_state=42, verbose=False)
    
    best_params['elasticity'] = result_a.x[0]
    best_params['mass'] = result_a.x[1]
    print(f"--- Stage A Complete. Best found: e={best_params['elasticity']:.4f}, m={best_params['mass']:.4f} ---")

    # === STAGE B: OPTIMIZE FRICTION ===
    current_stage = "Stage B (Slide)"
    call_count = 0
    print(f"\n--- {current_stage}: Isolating Friction ---")

    sim_b = Simulator(GROUND_TRUTH_PARAMS)
    traj_b = sim_b.run_simulation_for_trajectory(steps=EXP_B_STEPS, impulses=EXP_B_IMPULSES)
    
    search_space_b = [Real(0.1, 1.0, name='friction')]
    
    obj_b = partial(objective_function, 
                    search_space_names=['friction'],
                    fixed_params={'elasticity': best_params['elasticity'], 'mass': best_params['mass']},
                    ground_truth_traj=traj_b, steps=EXP_B_STEPS, impulses=EXP_B_IMPULSES)

    result_b = gp_minimize(func=obj_b, dimensions=search_space_b, n_calls=30, n_initial_points=15, random_state=42, verbose=False)
    
    best_params['friction'] = result_b.x[0]
    print(f"--- Stage B Complete. Best found: f={best_params['friction']:.4f} ---")

    # === STAGE C: FINAL REFINEMENT ===
    current_stage = "Stage C (Refine)"
    call_count = 0
    print(f"\n--- {current_stage}: Final Refinement of All Parameters ---")

    sim_c = Simulator(GROUND_TRUTH_PARAMS)
    traj_c = sim_c.run_simulation_for_trajectory(steps=EXP_C_STEPS, impulses=EXP_C_IMPULSES)
    
    search_space_c = [Real(0.1, 1.0, name='friction'), Real(0.1, 1.0, name='elasticity'), Real(5.0, 25.0, name='mass')]
    
    obj_c = partial(objective_function, 
                    search_space_names=['friction', 'elasticity', 'mass'],
                    fixed_params={},
                    ground_truth_traj=traj_c, steps=EXP_C_STEPS, impulses=EXP_C_IMPULSES)

    result_c = gp_minimize(func=obj_c, dimensions=search_space_c, n_calls=70, x0=list(best_params.values()), n_initial_points=35, random_state=42, verbose=False)
    
    final_params = {
        'friction': result_c.x[0],
        'elasticity': result_c.x[1],
        'mass': result_c.x[2]
    }
    print("--- Refinement Complete ---")

    # --- Final Results ---
    print("\n\n--- V2 - Definitive Calibration Results ---")
    print("\nComparison of Parameters:")
    print("                   | Friction | Elasticity | Mass")
    print("-------------------|----------|------------|-------")
    print(f"True Values        | {GROUND_TRUTH_PARAMS['friction']:.4f}   | {GROUND_TRUTH_PARAMS['elasticity']:.4f}     | {GROUND_TRUTH_PARAMS['mass']:.2f}")
    print(f"Calibrated Values  | {final_params['friction']:.4f}   | {final_params['elasticity']:.4f}     | {final_params['mass']:.2f}")
    print("-----------------------------------------------")

    # --- Generate Final Plot ---
    plot_results(
        ground_truth_traj=traj_c,
        initial_guess_params=initial_guess,
        final_params=final_params,
        impulses=EXP_C_IMPULSES,
        steps=EXP_C_STEPS,
        filename="paper/figures/trajectory_comparison.png"
    )


if __name__ == "__main__":
    run_staged_calibration()

