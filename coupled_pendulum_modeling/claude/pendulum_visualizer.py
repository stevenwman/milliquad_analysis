"""
Pendulum Visualization
Interactive animation and trajectory plots for the pendulum simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import pickle


class PendulumVisualizer:
    def __init__(self, data_file='pendulum_data.pkl'):
        """Load simulation data"""
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        self.t = data['t']
        self.y = data['y']
        self.params = data['params']
        self.l = self.params['l']
        self.wb = self.params['wb']
        
        print(f'Loaded data: {len(self.t)} time points, duration = {self.t[-1]:.4f} s')
    
    def create_interactive_animation(self, fps=30):
        """Create interactive animation with slider and play button"""
        # Interpolate data for smooth playback
        num_frames = int(self.t[-1] * fps)
        self.time_vector = np.linspace(self.t[0], self.t[-1], num_frames)
        self.theta1_interp = np.interp(self.time_vector, self.t, self.y[:, 0])
        self.theta2_interp = np.interp(self.time_vector, self.t, self.y[:, 2])
        self.num_frames = num_frames
        self.fps = fps
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(8, 9))
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.7])
        
        # Setup plot
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        self.pivot1_x = -1.5 * self.l
        self.pivot2_x = 1.5 * self.l
        pivot_y = 0
        plot_width = 4 * self.l
        
        self.ax.set_xlim(-plot_width, plot_width)
        self.ax.set_ylim(-1.5 * self.l, 1.5 * self.l)
        self.ax.set_xlabel('x-position (m)')
        self.ax.set_ylabel('y-position (m)')
        
        # Initialize plot objects
        self.rod1, = self.ax.plot([self.pivot1_x, self.pivot1_x], [pivot_y, -self.l], 
                                   'r-', linewidth=2, label='Pendulum 1')
        self.bob1, = self.ax.plot(self.pivot1_x, -self.l, 'ro', markersize=15)
        
        self.rod2, = self.ax.plot([self.pivot2_x, self.pivot2_x], [pivot_y, -self.l], 
                                   'b-', linewidth=2, label='Pendulum 2')
        self.bob2, = self.ax.plot(self.pivot2_x, -self.l, 'bo', markersize=15)
        
        self.time_title = self.ax.set_title('Time: 0.000 s')
        
        # Add rotating magnetic field arrows
        arrow_length = 0.8 * self.l
        self.arrow1 = self.ax.quiver(self.pivot1_x, 0, arrow_length, 0, 
                                      color='green', width=0.006, scale=1, scale_units='xy')
        self.arrow2 = self.ax.quiver(self.pivot2_x, 0, arrow_length, 0, 
                                      color='green', width=0.006, scale=1, scale_units='xy')
        
        self.arrow_length = arrow_length
        self.ax.legend(loc='upper right')
        
        # Create slider
        ax_slider = self.fig.add_axes([0.15, 0.08, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, 
                             valinit=0, valstep=1)
        self.slider.on_changed(self.update_frame)
        
        # Create play button
        ax_button = self.fig.add_axes([0.42, 0.02, 0.15, 0.04])
        self.button = Button(ax_button, 'Play')
        self.button.on_clicked(self.play_pause)
        
        self.playing = False
        self.anim = None
        
        plt.show()
    
    def update_frame(self, val):
        """Update the animation to the current frame"""
        frame_idx = int(self.slider.val)
        
        th1 = self.theta1_interp[frame_idx]
        th2 = self.theta2_interp[frame_idx]
        
        # Calculate bob positions
        x_bob1 = self.pivot1_x + self.l * np.sin(th1)
        y_bob1 = 0 - self.l * np.cos(th1)
        x_bob2 = self.pivot2_x + self.l * np.sin(th2)
        y_bob2 = 0 - self.l * np.cos(th2)
        
        # Update plot objects
        self.rod1.set_data([self.pivot1_x, x_bob1], [0, y_bob1])
        self.bob1.set_data([x_bob1], [y_bob1])
        self.rod2.set_data([self.pivot2_x, x_bob2], [0, y_bob2])
        self.bob2.set_data([x_bob2], [y_bob2])
        
        self.time_title.set_text(f'Time: {self.time_vector[frame_idx]:.3f} s')
        
        # Update rotating magnetic field
        t_current = self.time_vector[frame_idx]
        angle = self.wb * t_current
        u = self.arrow_length * np.sin(angle)
        v = self.arrow_length * -np.cos(angle)
        
        self.arrow1.set_UVC(u, v)
        self.arrow2.set_UVC(u, v)
        
        self.fig.canvas.draw_idle()
    
    def play_pause(self, event):
        """Toggle play/pause"""
        if not self.playing:
            self.playing = True
            self.button.label.set_text('Pause')
            self.anim = FuncAnimation(self.fig, self.animate, 
                                     frames=range(int(self.slider.val), self.num_frames),
                                     interval=1000/self.fps, repeat=False, blit=False)
            plt.draw()
        else:
            self.playing = False
            self.button.label.set_text('Play')
            if self.anim:
                self.anim.event_source.stop()
    
    def animate(self, frame):
        """Animation function"""
        if not self.playing:
            return
        
        self.slider.set_val(frame)
        if frame >= self.num_frames - 1:
            self.playing = False
            self.button.label.set_text('Play')
            if self.anim:
                self.anim.event_source.stop()
    
    def plot_trajectories(self):
        """Plot trajectory analysis figures"""
        # Figure 1: Time series
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1_left = ax1
        ax1_right = ax1.twinx()
        
        # Left axis: velocities
        ax1_left.plot(self.t, self.y[:, 1], label='θ1_dot', color='C0')
        ax1_left.plot(self.t, self.y[:, 3], label='θ2_dot', color='C1')
        ax1_left.set_ylabel('Joint velocity [rad/s]', color='C0')
        ax1_left.tick_params(axis='y', labelcolor='C0')
        ax1_left.legend(loc='upper left')
        
        # Right axis: angles (wrapped to [0, 2π])
        theta1_wrapped = np.mod(self.y[:, 0], 2*np.pi)
        theta2_wrapped = np.mod(self.y[:, 2], 2*np.pi)
        ax1_right.plot(self.t, theta1_wrapped, label='θ1', color='C2', linestyle='--')
        ax1_right.plot(self.t, theta2_wrapped, label='θ2', color='C3', linestyle='--')
        ax1_right.set_ylabel('Joint angle [rad]', color='C2')
        ax1_right.tick_params(axis='y', labelcolor='C2')
        ax1_right.legend(loc='upper right')
        
        ax1.set_xlabel('Time [s]')
        ax1.set_title('Pendulum State Variables vs Time')
        ax1.grid(True, alpha=0.3)
        
        # Figure 2: Phase portrait (polar)
        fig2, ax2 = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
        ax2.plot(self.y[:, 0], self.y[:, 1], label='Pendulum 1')
        ax2.set_title('Phase Portrait: θ vs θ_dot')
        ax2.legend()
        
        # Figure 3: Resonance force
        fig3, ax3 = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
        Ff = self.params['Ff']
        x_range = self.params['x_range']
        x_offset = self.params['x_offset']
        F_plot = -Ff * np.exp(-((self.y[:, 0] - x_offset) * x_range)**2) * self.y[:, 1]
        ax3.plot(self.y[:, 0], F_plot)
        ax3.set_title('Resonance Force vs Angle')
        
        plt.show()
    
    def save_animation(self, filename='pendulum_animation.mp4', duration=10, fps=30):
        """Save animation as MP4 video"""
        print(f'Creating animation video: {duration}s at {fps} fps...')
        
        # Interpolate data for video
        num_frames = int(duration * fps)
        time_vector = np.linspace(self.t[0], self.t[-1], num_frames)
        theta1_interp = np.interp(time_vector, self.t, self.y[:, 0])
        theta2_interp = np.interp(time_vector, self.t, self.y[:, 2])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.grid(True)
        ax.set_aspect('equal')
        
        pivot1_x = -1.5 * self.l
        pivot2_x = 1.5 * self.l
        plot_width = 4 * self.l
        
        ax.set_xlim(-plot_width, plot_width)
        ax.set_ylim(-1.5 * self.l, 1.5 * self.l)
        ax.set_xlabel('x-position (m)')
        ax.set_ylabel('y-position (m)')
        
        # Initialize plot objects
        rod1, = ax.plot([], [], 'r-', linewidth=2, label='Pendulum 1')
        bob1, = ax.plot([], [], 'ro', markersize=15)
        rod2, = ax.plot([], [], 'b-', linewidth=2, label='Pendulum 2')
        bob2, = ax.plot([], [], 'bo', markersize=15)
        time_text = ax.set_title('')
        
        arrow_length = 0.8 * self.l
        arrow1 = ax.quiver(pivot1_x, 0, arrow_length, 0, 
                          color='green', width=0.006, scale=1, scale_units='xy')
        arrow2 = ax.quiver(pivot2_x, 0, arrow_length, 0, 
                          color='green', width=0.006, scale=1, scale_units='xy')
        ax.legend(loc='upper right')
        
        def animate_video(frame):
            th1 = theta1_interp[frame]
            th2 = theta2_interp[frame]
            
            x_bob1 = pivot1_x + self.l * np.sin(th1)
            y_bob1 = 0 - self.l * np.cos(th1)
            x_bob2 = pivot2_x + self.l * np.sin(th2)
            y_bob2 = 0 - self.l * np.cos(th2)
            
            rod1.set_data([pivot1_x, x_bob1], [0, y_bob1])
            bob1.set_data([x_bob1], [y_bob1])
            rod2.set_data([pivot2_x, x_bob2], [0, y_bob2])
            bob2.set_data([x_bob2], [y_bob2])
            
            time_text.set_text(f'Time: {time_vector[frame]:.3f} s')
            
            t_current = time_vector[frame]
            angle = self.wb * t_current
            u = arrow_length * np.sin(angle)
            v = arrow_length * -np.cos(angle)
            arrow1.set_UVC(u, v)
            arrow2.set_UVC(u, v)
            
            return rod1, bob1, rod2, bob2, time_text, arrow1, arrow2
        
        anim = FuncAnimation(fig, animate_video, frames=num_frames, 
                           interval=1000/fps, blit=True)
        
        # Save the animation
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(filename, writer=writer)
            print(f'Animation saved to {filename}')
        except Exception as e:
            print(f'Error saving animation: {e}')
            print('Make sure ffmpeg is installed: pip install ffmpeg-python')
        
        plt.close(fig)


def main():
    """Main visualization function"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'interactive'
    
    viz = PendulumVisualizer()
    
    if mode == 'interactive' or mode == 'i':
        print('Starting interactive animation...')
        viz.create_interactive_animation(fps=30)
    elif mode == 'plot' or mode == 'p':
        print('Creating trajectory plots...')
        viz.plot_trajectories()
    elif mode == 'video' or mode == 'v':
        print('Saving animation video...')
        viz.save_animation('pendulum_animation.mp4', duration=10, fps=30)
        print('Done!')
    elif mode == 'all':
        print('Creating all visualizations...')
        viz.plot_trajectories()
        viz.create_interactive_animation(fps=30)
    else:
        print('Usage: python pendulum_visualizer.py [mode]')
        print('Modes:')
        print('  interactive (i) - Interactive animation with slider (default)')
        print('  plot (p)        - Trajectory plots')
        print('  video (v)       - Save animation as MP4')
        print('  all             - All plots and interactive animation')


if __name__ == '__main__':
    main()