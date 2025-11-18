import csv
import argparse
import sim_optimizer as sim_optimizer

def main():
    """
    Reads a specific result from the optimization_results.csv file and
    launches a visualized MuJoCo simulation with its parameters, with an
    option to record a video instead.
    """
    parser = argparse.ArgumentParser(
        description="Visualize or record a rollout from optimization_results.csv."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="The rank of the result to visualize (1 is the best)."
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional. Path to save a video recording, e.g., 'rollout.mp4'."
    )
    args = parser.parse_args()

    # --- 1. Read the CSV file ---
    try:
        with open('optimization_results.csv', 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
    except FileNotFoundError:
        print("Error: optimization_results.csv not found.")
        print("Please run tune_params.py first.")
        return

    if not results or args.rank > len(results):
        print(f"Error: Rank {args.rank} is out of bounds.")
        print(f"The CSV file only contains {len(results)} results.")
        return

    # --- 2. Select the desired result ---
    # The CSV is already sorted, so we can just index it.
    selected_run = results[args.rank - 1]

    print(f"--- Visualizing Rank #{args.rank} ---")
    print(f"  ID: {selected_run['id']}")
    print(f"  Cost: {float(selected_run['cost']):.6f}")
    print(f"  Avg Velocity: {float(selected_run['avg_velocity']):.4f} m/s")

    # --- 3. Reconstruct the simulation parameters ---
    try:
        sim_params = {
            'ground_friction': [
                float(selected_run['sliding_friction']),
                float(selected_run['torsional_friction']),
                float(selected_run['rolling_friction'])
            ],
            'solref': [
                float(selected_run['solref_timeconst']),
                float(selected_run['solref_dampratio'])
            ],
            'solimp': [
                float(selected_run['solimp_dmin']),
                float(selected_run['solimp_dmax']),
                float(selected_run['solimp_width']),
                0.5,
                1.0
            ],
            # Use the same constant values as the optimizer
            'dof_damping': 7e-10,
            'kp_mag': 2.5e-6
        }
    except KeyError as e:
        print(f"Error: Missing parameter {e} in the CSV file.")
        return
    except ValueError as e:
        print(f"Error: Could not parse parameter values in the CSV file: {e}")
        return


    filename = "mulit_milli_quad/scene_2.xml"
    # --- 4. Run the simulation with visualization or recording ---
    if args.record:
        print(f"\nRecording rollout to {args.record}...")
        sim_optimizer.run_simulation(
            sim_params,
            sim_duration=10.0,
            record_path=args.record,
            ignore_stuck_detection=True
        )
    else:
        print("\nLaunching simulation... (Press SPACE to play/pause)")
        sim_optimizer.run_simulation(
            sim_params,
            mjcf_path=filename,
            sim_duration=10.0, # Longer duration for viewing
            visualize=True,
            ignore_stuck_detection=True
        )

if __name__ == "__main__":
    main()
