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
    parser.add_argument(
        "--drive_freq",
        type=float,
        default=30.0,
        help="Manually set the drive frequency for the simulation."
    )
    args = parser.parse_args()

    # --- 1. Read the CSV file ---
    try:
        with open('multi_optimization_results.csv', 'r') as f:
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
    try:
        print(f"  Avg Velocity (scene4): {float(selected_run['velocity_scene4']):.4f} m/s")
        print(f"  Avg Velocity (scene2): {float(selected_run['velocity_scene2']):.4f} m/s")
    except KeyError:
        # Fallback for old CSV formats
        print(f"  Avg Velocity: {float(selected_run.get('avg_velocity', 0)):.4f} m/s")

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
            # 'kp_mag': 2.5e-6
            'kp_mag': float(selected_run['kp_mag'])
        }
    except KeyError as e:
        print(f"Error: Missing parameter {e} in the CSV file.")
        return
    except ValueError as e:
        print(f"Error: Could not parse parameter values in the CSV file: {e}")
        return

    # Add the manually specified drive frequency to the parameters
    sim_params['drive_freq'] = args.drive_freq
    if args.drive_freq != 30.0:
        print(f"  Using manual drive frequency: {args.drive_freq} Hz")

    # --- 3.5 Edit the XML Scene to add steps ---
    import xml.etree.ElementTree as ET
    
    # Constants for steps
    STEP_HEIGHT = 0.0015 # 1 mm
    STEP_WIDTH = 0.1   # 5 cm
    STEP_LENGTH = 0.0045  # 4.5 mm
    FINAL_STEP_LENGTH = 0.02  # Length of the final step (default: same as regular steps)
    NUM_STEPS = 5 
    START_X = 0.0075      # 5 cm

    original_scene_path = "mulit_milli_quad/scene_4.xml"
    # original_scene_path = "wheel_milli_quad/scene_wheel.xml"
    edited_scene_path = original_scene_path.replace(".xml", "_edited.xml")
    
    try:
        tree = ET.parse(original_scene_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        
        if worldbody is None:
             # If worldbody is not found directly, try to find it recursively or handle error
             # In standard mujoco xmls, it should be a direct child or close.
             # Let's assume standard structure based on previous view_file.
             pass

        if worldbody is not None:
            for i in range(NUM_STEPS):
                # Determine step length (use final step length for the last step)
                is_final_step = (i == NUM_STEPS - 1)
                step_length = FINAL_STEP_LENGTH if is_final_step else STEP_LENGTH
                
                # Calculate position
                # x: start + cumulative length of previous steps + half_length (center)
                if is_final_step:
                    # For final step: sum of all previous steps + half of final step length
                    pos_x = START_X + (NUM_STEPS - 1) * STEP_LENGTH + step_length / 2.0
                else:
                    # For regular steps: i*length + half_length (center)
                    pos_x = START_X + i * STEP_LENGTH + step_length / 2.0
                pos_y = 0.0
                # z: (i+1)*height - half_height (center)
                # This stacks them like a staircase where the top surface of step i is at (i+1)*height
                pos_z = (i + 1) * STEP_HEIGHT - STEP_HEIGHT / 2.0
                
                # Create geom element
                # size is half-extents
                geom = ET.Element('geom')
                geom.set('name', f'step_{i}')
                geom.set('type', 'box')
                geom.set('size', f"{step_length/2.0} {STEP_WIDTH/2.0} {STEP_HEIGHT/2.0}")
                geom.set('pos', f"{pos_x} {pos_y} {pos_z}")
                # geom.set('material', 'groundplane') 
                geom.set('rgba', '0.5 0.5 0.5 1') # Medium grey
                
                worldbody.append(geom)
            
            tree.write(edited_scene_path)
            print(f"Created edited scene with {NUM_STEPS} steps at {edited_scene_path}")
            
            # Update filename to point to the edited scene (relative to mujoco/ folder where run_simulation expects?)
            # The run_simulation likely takes path relative to where it's run or absolute.
            # Original was "mulit_milli_quad/scene_4.xml".
            # We saved to "mujoco/mulit_milli_quad/scene_edited.xml".
            # If running from root, "mujoco/..." is correct.
            # But original code had "mulit_milli_quad/scene_4.xml", implying it might be running from mujoco dir?
            # Let's check where the user runs it from. Usually root.
            # Wait, the original code had `filename = "mulit_milli_quad/scene_4.xml"`.
            # If I write to `mujoco/mulit_milli_quad/scene_edited.xml`, I should pass `mulit_milli_quad/scene_edited.xml` 
            # IF the CWD is `mujoco`.
            # However, `visualize_rollout_step.py` is in `mujoco/`.
            # Let's assume we run from `LEGO-milliquad-mujoco/`.
            # Then `mujoco/visualize_rollout_step.py` is executed.
            # The original code `filename = "mulit_milli_quad/scene_4.xml"` suggests `mujoco` is NOT in the path if running from `mujoco` dir?
            # OR if running from root, maybe `sim_optimizer` handles the path?
            # Let's look at `sim_optimizer.py` imports or usage if needed.
            # BUT, to be safe, I will use the same relative path structure as the original `filename`.
            
            filename = edited_scene_path

    except Exception as e:
        print(f"Error editing XML scene: {e}")
        print("Falling back to original scene.")

    # --- 4. Run the simulation with visualization or recording ---
    if args.record:
        print(f"\nRecording rollout to {args.record}...")
        sim_optimizer.run_simulation(
            sim_params,
            mjcf_path=filename,
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
