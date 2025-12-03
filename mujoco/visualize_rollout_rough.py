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

    # --- 3.5 Edit the XML Scene to add rough terrain ---
    import xml.etree.ElementTree as ET
    import random
    import numpy as np
    
    # Constants for rough terrain
    HEIGHT_RANGE = 0.001 # +/- 1 mm
    SEED = 42
    SQUARE_SIZE_X = 0.005 # Keep user's 5mm
    SQUARE_SIZE_Y = 0.005 # Keep user's 5mm
    NUM_X = 20
    NUM_Y = 10
    START_X = 0.05      # 5 cm
    
    random.seed(SEED)

    original_scene_path = "mulit_milli_quad/scene_1.xml"
    edited_scene_path = original_scene_path.replace(".xml", "_edited_rough.xml")
    
    try:
        tree = ET.parse(original_scene_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        
        if worldbody is None:
             pass

        if worldbody is not None:
            # 1. Modify the floor to stop at START_X
            floor = None
            for geom in worldbody.findall('geom'):
                if geom.get('name') == 'floor':
                    floor = geom
                    break
            
            if floor is not None:
                floor_len = 1.0
                floor_center_x = START_X - floor_len
                floor.set('pos', f"{floor_center_x} 0 0")
                floor.set('size', f"{floor_len} 1.0 0.05")
                floor.set('type', 'box')
                floor.set('pos', f"{floor_center_x} 0 -0.05")
            
            # 2. Generate Rough Terrain using HField
            # Generate heightmap image
            import imageio
            
            # To achieve "blocky" look, we need higher resolution.
            PIXELS_PER_SQUARE = 20
            
            # Generate random heights for each logical square
            # Uniform random in [-HEIGHT_RANGE, HEIGHT_RANGE]
            logical_heights = np.zeros((NUM_Y, NUM_X))
            for ix in range(NUM_X):
                for iy in range(NUM_Y):
                    logical_heights[iy, ix] = random.uniform(-HEIGHT_RANGE, HEIGHT_RANGE)
            
            # Expand to full resolution image
            heights = np.kron(logical_heights, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE)))
            
            # Normalize to 0-255
            # Range is [-HEIGHT_RANGE, HEIGHT_RANGE]
            # Map -HEIGHT_RANGE -> 0
            # Map HEIGHT_RANGE -> 255
            # Total span = 2 * HEIGHT_RANGE
            
            z_min = -HEIGHT_RANGE
            z_span = 2 * HEIGHT_RANGE
            
            normalized_heights = (heights - z_min) / z_span
            # Clip to 0-1 just in case floating point issues
            normalized_heights = np.clip(normalized_heights, 0.0, 1.0)
            
            # Convert to 0-255 uint8
            img_data = (normalized_heights * 255).astype(np.uint8)
            
            # Save as PNG
            hfield_filename = "mulit_milli_quad/rough_heightmap.png"
            imageio.imwrite(hfield_filename, img_data)
            print(f"Created heightmap image at {hfield_filename}")
            
            # Add asset and geom to XML
            asset = root.find('asset')
            if asset is None:
                asset = ET.SubElement(root, 'asset')
            
            import os
            abs_hfield_path = os.path.abspath(hfield_filename)
            
            hfield_asset = ET.SubElement(asset, 'hfield')
            hfield_asset.set('name', 'rough_terrain')
            hfield_asset.set('file', abs_hfield_path)
            
            x_half = (NUM_X * SQUARE_SIZE_X) / 2.0
            y_half = (NUM_Y * SQUARE_SIZE_Y) / 2.0
            
            # size: x_half y_half z_scale z_base
            # z_scale = z_span (total variation)
            # z_base = z_min (offset from 0)
            # Mujoco requires positive size parameters.
            # We set z_base to a small positive value and compensate with pos_z.
            z_base_safe = 0.001
            hfield_asset.set('size', f"{x_half} {y_half} {z_span} {z_base_safe}") 
            
            # Geom
            pos_x = START_X + x_half
            pos_y = 0.0
            
            # We want the surface to range from -HEIGHT_RANGE to HEIGHT_RANGE.
            # With z_base_safe, the surface ranges from [z_base_safe, z_base_safe + z_span].
            # i.e. [0.001, 0.001 + 0.002] = [0.001, 0.003].
            # We want [-0.001, 0.001].
            # So we need to shift down by 0.002.
            pos_z = -0.002
            
            hfield_geom = ET.SubElement(worldbody, 'geom')
            hfield_geom.set('name', 'rough_terrain_geom')
            hfield_geom.set('type', 'hfield')
            hfield_geom.set('hfield', 'rough_terrain')
            hfield_geom.set('pos', f"{pos_x} {pos_y} {pos_z}")
            hfield_geom.set('rgba', '0.5 0.5 0.5 1')
            
            tree.write(edited_scene_path)
            print(f"Created rough terrain scene at {edited_scene_path}")
            
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
