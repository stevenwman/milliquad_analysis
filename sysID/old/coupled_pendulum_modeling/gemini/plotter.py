# plotter.py
import pickle
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, FFMpegWriter

def plot_static(sol, params):
    """
    Generates the static plots from the simulation data.
    """
    t = sol.t
    y = sol.y.T  # Transpose to get (n_times, n_states)
    
    Ff = params['Ff']
    x_offset = params['x_offset']
    x_range = params['x_range']

    # --- Plot 1: Time History (Velocities and Angles) ---
    fig1, ax1 = plt.subplots()
    
    # Plot velocities on the left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('joint velocity [rad/s]', color=color1)
    ax1.plot(t, y[:, 1], color=color1, linestyle='-', label='x2 (vel1)')
    ax1.plot(t, y[:, 3], color=color1, linestyle='--', label='x4 (vel2)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create a second y-axis for angles
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('joint angle [rad]', color=color2)
    # Use (angle % (2*pi)) for wrapTo2Pi
    ax2.plot(t, y[:, 0] % (2 * np.pi), color=color2, linestyle='-', label='x1 (ang1)')
    ax2.plot(t, y[:, 2] % (2 * np.pi), color=color2, linestyle='--', label='x3 (ang2)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig1.suptitle('State vs. Time')
    fig1.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Plot 2: Phase Portrait (Polar) ---
    fig2 = plt.figure()
    ax_polar = fig2.add_subplot(111, projection='polar')
    ax_polar.plot(y[:, 0], y[:, 1], label='Pendulum 1')
    ax_polar.plot(y[:, 2], y[:, 3], label='Pendulum 2', linestyle='--')
    ax_polar.set_title('Phase Portrait (Angle vs. Velocity)')
    ax_polar.legend()
    
    # --- Plot 3: Resistive Force (Polar) ---
    F_plot = -Ff * np.exp(-((y[:, 0] - x_offset) * x_range)**2) * y[:, 1]
    
    fig3 = plt.figure()
    ax_force_polar = fig3.add_subplot(111, projection='polar')
    ax_force_polar.plot(y[:, 0], F_plot)
    ax_force_polar.set_title('Resistive Force vs. Angle (Pendulum 1)')

def run_interactive_animation(sol, params):
    """
    Launches an interactive animation player with a slider and play/pause button.
    """
    print("Building interactive player...")
    
    # --- 1. Interpolate data ---
    # Use the 'sol' object (from dense_output=True) to get a smooth solution
    fps = 60  # A more reasonable playback speed
    t_start, t_end = sol.t[0], sol.t[-1]
    num_frames = int(np.ceil(t_end * fps))
    time_vector = np.linspace(t_start, t_end, num_frames)
    
    # sol.sol() is the interpolating function
    interp_states = sol.sol(time_vector)
    theta1_interp = interp_states[0, :]
    theta2_interp = interp_states[2, :]
    
    l = params['l']
    wb = params['wb']

    # --- 2. Create the UI Figure and Axes ---
    fig = plt.figure('Interactive Pendulum Playback', figsize=(6, 7))
    # Main axes for animation
    ax = fig.add_axes([0.1, 0.25, 0.8, 0.7])
    
    # --- 3. Setup the initial plot ---
    ax.set_aspect('equal')
    ax.grid(True)
    pivot1_x = -1.5 * l
    pivot2_x = 1.5 * l
    pivot_y = 0
    plot_width = 4 * l
    ax.set_xlim(-plot_width, plot_width)
    ax.set_ylim(-1.5 * l, 1.5 * l)
    ax.set_xlabel('x-position (m)')
    ax.set_ylabel('y-position (m)')

    # Initialize plot artists (lines, markers, arrows)
    rod1, = ax.plot([], [], 'r-', lw=2)
    bob1, = ax.plot([], [], 'ro', ms=15, mfc='r')
    rod2, = ax.plot([], [], 'b-', lw=2)
    bob2, = ax.plot([], [], 'bo', ms=15, mfc='b')
    time_title = ax.set_title('')
    
    arrow_length = 0.8 * l
    # Create quiver (arrow) objects
    mag_field_arrow1 = ax.quiver(pivot1_x, 0, arrow_length, 0,
                                 color=[0.1, 0.7, 0.1], scale=1, 
                                 scale_units='xy', angles='xy')
    mag_field_arrow2 = ax.quiver(pivot2_x, 0, arrow_length, 0,
                                 color=[0.1, 0.7, 0.1], scale=1, 
                                 scale_units='xy', angles='xy')
                                 
    # --- 4. Create UI Controls ---
    # Axes for slider and button
    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    ax_button = fig.add_axes([0.45, 0.12, 0.1, 0.04])
    
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    button = Button(ax_button, 'Play')

    # --- 5. Define Update and Callback Functions ---
    def update_plot(frame_idx):
        """Updates all artists for a given frame index."""
        frame_idx = int(frame_idx) # Ensure it's an integer
        
        th1 = theta1_interp[frame_idx]
        th2 = theta2_interp[frame_idx]
        
        x_bob1 = pivot1_x + l * np.sin(th1)
        y_bob1 = pivot_y - l * np.cos(th1)
        x_bob2 = pivot2_x + l * np.sin(th2)
        y_bob2 = pivot_y - l * np.cos(th2)
        
        rod1.set_data([pivot1_x, x_bob1], [pivot_y, y_bob1])
        bob1.set_data([x_bob1], [y_bob1])  # <-- Added brackets
        rod2.set_data([pivot2_x, x_bob2], [pivot_y, y_bob2])
        bob2.set_data([x_bob2], [y_bob2])  # <-- Added brackets
        
        t_current = time_vector[frame_idx]
        time_title.set_text(f'Time: {t_current:.3f} s')
        
        # Update magnetic field arrows
        angle = wb * t_current
        u = arrow_length * np.sin(angle)
        v = arrow_length * -np.cos(angle)
        mag_field_arrow1.set_UVC(u, v)
        mag_field_arrow2.set_UVC(u, v)
        
        fig.canvas.draw_idle()

    # Link slider to the update function
    slider.on_changed(update_plot)
    
    # This class manages the play/pause state
    class Player:
        def __init__(self):
            self.playing = False
        
        def toggle_play(self, event):
            if self.playing:
                self.pause()
            else:
                self.play()
                
        def pause(self):
            self.playing = False
            button.label.set_text('Play')

        def play(self):
            self.playing = True
            button.label.set_text('Pause')
            current_frame = int(slider.val)
            
            # Loop for playing animation
            while self.playing and current_frame < num_frames - 1:
                current_frame += 1
                slider.set_val(current_frame) # This triggers update_plot
                # Use plt.pause for a non-blocking draw/delay
                plt.pause(1.0 / fps)
                # Check state again in case pause was clicked
                if not self.playing:
                    break
            
            self.pause() # Reset button when done or paused
            
    player = Player()
    button.on_clicked(player.toggle_play)

    # Initialize the plot to the first frame
    update_plot(0)


def record_video(sol, params):
    """
    Records the animation to an MP4 video file using FuncAnimation.
    """
    print("Recording animation to video...")
    
    # --- 1. Define Video Export Settings ---
    output_duration_seconds = 10
    output_fps = 30
    video_filename = 'pendulum_playback.mp4'

    # --- 2. Generate and Interpolate Data ---
    num_video_frames = round(output_duration_seconds * output_fps)
    t_start, t_end = sol.t[0], sol.t[-1]
    video_time_vector = np.linspace(t_start, t_end, num_video_frames)
    
    video_states = sol.sol(video_time_vector)
    video_theta1 = video_states[0, :]
    video_theta2 = video_states[2, :]
    
    l = params['l']
    wb = params['wb']

    # --- 3. Set up the Figure and Artists (for recording) ---
    fig_rec = plt.figure(figsize=(6, 7))
    ax_rec = fig_rec.add_axes([0.1, 0.1, 0.8, 0.8])
    
    ax_rec.set_aspect('equal')
    ax_rec.grid(True)
    pivot1_x = -1.5 * l
    pivot2_x = 1.5 * l
    pivot_y = 0
    plot_width = 4 * l
    ax_rec.set_xlim(-plot_width, plot_width)
    ax_rec.set_ylim(-1.5 * l, 1.5 * l)
    ax_rec.set_xlabel('x-position (m)')
    ax_rec.set_ylabel('y-position (m)')

    rod1_rec, = ax_rec.plot([], [], 'r-', lw=2)
    bob1_rec, = ax_rec.plot([], [], 'ro', ms=15, mfc='r')
    rod2_rec, = ax_rec.plot([], [], 'b-', lw=2)
    bob2_rec, = ax_rec.plot([], [], 'bo', ms=15, mfc='b')
    time_title_rec = ax_rec.set_title('')
    
    arrow_length = 0.8 * l
    mag_arrow1_rec = ax_rec.quiver(pivot1_x, 0, arrow_length, 0,
                                   color=[0.1, 0.7, 0.1], scale=1, 
                                   scale_units='xy', angles='xy')
    mag_arrow2_rec = ax_rec.quiver(pivot2_x, 0, arrow_length, 0,
                                   color=[0.1, 0.7, 0.1], scale=1, 
                                   scale_units='xy', angles='xy')

    # --- 4. Define the Animation Update Function ---
    def update_video(frame_idx):
        th1 = video_theta1[frame_idx]
        th2 = video_theta2[frame_idx]
        
        x_bob1 = pivot1_x + l * np.sin(th1)
        y_bob1 = pivot_y - l * np.cos(th1)
        x_bob2 = pivot2_x + l * np.sin(th2)
        y_bob2 = pivot_y - l * np.cos(th2)
        
        rod1_rec.set_data([pivot1_x, x_bob1], [pivot_y, y_bob1])
        bob1_rec.set_data([x_bob1], [y_bob1])  # <-- Added brackets
        rod2_rec.set_data([pivot2_x, x_bob2], [pivot_y, y_bob2])
        bob2_rec.set_data([x_bob2], [y_bob2])  # <-- Added brackets
        
        t_current = video_time_vector[frame_idx]
        time_title_rec.set_text(f'Time: {t_current:.3f} s')
        
        angle = wb * t_current
        u = arrow_length * np.sin(angle)
        v = arrow_length * -np.cos(angle)
        mag_arrow1_rec.set_UVC(u, v)
        mag_arrow2_rec.set_UVC(u, v)
        
        # Return all modified artists
        return (rod1_rec, bob1_rec, rod2_rec, bob2_rec, 
                time_title_rec, mag_arrow1_rec, mag_arrow2_rec)

    # --- 5. Create and Save the Animation ---
    ani = FuncAnimation(
        fig_rec, 
        update_video, 
        frames=num_video_frames, 
        blit=False  # Blit=False is often more reliable with quiver
    )
    
    # Setup the writer
    writer = FFMpegWriter(fps=output_fps)
    
    # Save the animation
    ani.save(video_filename, writer=writer)
    
    plt.close(fig_rec) # Close the figure
    print(f'Video saved as {video_filename}')


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot and animate pendulum simulation results."
    )
    parser.add_argument(
        '--static', 
        action='store_true', 
        help='Show static plots.'
    )
    parser.add_argument(
        '--animate', 
        action='store_true', 
        help='Run interactive animation.'
    )
    parser.add_argument(
        '--record', 
        action='store_true', 
        help='Record animation to MP4 (requires ffmpeg).'
    )
    args = parser.parse_args()

    # --- Load Data ---
    data_filename = 'simulation_data.pkl'
    print(f"Loading simulation data from {data_filename}...")
    try:
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        sol = data['sol']
        params = data['params']
    except FileNotFoundError:
        print(f"Error: {data_filename} not found.", file=sys.stderr)
        print("Please run solver.py first to generate the data.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # If no flags are given, run all actions
    if not (args.static or args.animate or args.record):
        print("No specific action requested. Running all...")
        args.static = True
        args.animate = True
        args.record = True

    # --- Execute requested actions ---
    if args.record:
        try:
            record_video(sol, params)
        except FileNotFoundError:
            print("\nError: 'ffmpeg' not found.", file=sys.stderr)
            print("Video recording requires ffmpeg to be installed and in your system's PATH.", file=sys.stderr)
        except Exception as e:
            print(f"\nError during video recording: {e}", file=sys.stderr)
    
    if args.static:
        print("Generating static plots...")
        plot_static(sol, params)
        
    if args.animate:
        run_interactive_animation(sol, params)

    # Show all created figures (static and/or interactive)
    if args.static or args.animate:
        print("Displaying plots. Close all plot windows to exit.")
        plt.show()
    else:
        print("All tasks complete.")