import sim_optimizer as sim_optimizer
import numpy as np

def main():
    """
    Runs the simulation with a set of "insane" parameters known
    to cause instability, allowing for direct debugging of the
    simulation behavior when it hangs or crashes.
    """

    print("--- Preparing to run simulation with insane parameters ---")
    
    # These parameters are chosen to create a very stiff and under-damped
    # contact model, which is likely to cause the simulation to explode
    # or get stuck in a deadlock.
    insane_params = {
        'ground_friction': [1.0, 1.0, 1.0],
        'solref': [0.00001, 0.00001],  # Very small time constant and damping ratio
        'solimp': [0.999, 0.9999, 0.000001, 0.5, 1.0], # Stiff, narrow contact
        'dof_damping': 0.0,
        'kp_mag': 2.5e-6
    }

    print("Parameters being used:")
    for key, value in insane_params.items():
        print(f"  {key}: {value}")
    
    print("\nStarting simulation... If the script hangs here, it confirms a deadlock.")
    
    try:
        trajectory = sim_optimizer.run_simulation(
            insane_params, 
            sim_duration=15.0, 
            visualize=False
        )
        
        if trajectory is None:
            print("\nSimulation finished and correctly returned None (failure).")
        else:
            print(f"\nSimulation finished. Final time: {trajectory[-1]['time']:.4f}s")
            
    except Exception as e:
        print(f"\nSimulation threw an unhandled exception: {e}")

if __name__ == "__main__":
    main()
