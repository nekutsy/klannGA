import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def plot_multiple_trajectories(trajectories, colors=None, labels=None, title="Trajectory Comparison"):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    all_points = np.vstack(trajectories)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        label = labels[i] if labels else f'Simulation {i+1}'
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linewidth=2, label=label)
    
    if labels:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def animate_system(system, trajectory, steps=400, interval=50):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_points = np.vstack([p.position() for p in system.points] + [trajectory])
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Rod System Animation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    rods_lines = []
    for rod in system.rods:
        line, = ax.plot([], [], 'b-', linewidth=2)
        rods_lines.append(line)
    
    points_scatter = ax.scatter([], [], c='r', s=50)
    trajectory_line, = ax.plot([], [], 'g-', alpha=0.5)
    current_point, = ax.plot([], [], 'go', markersize=8)
    
    trajectory_x, trajectory_y = trajectory[:, 0], trajectory[:, 1]
    
    initial_points = [(p.x, p.y) for p in system.points]
    
    def init():
        for i, p in enumerate(system.points):
            p.x, p.y = initial_points[i]
            
        for line in rods_lines:
            line.set_data([], [])
        points_scatter.set_offsets(np.empty((0, 2)))
        trajectory_line.set_data([], [])
        current_point.set_data([], [])
        return rods_lines + [points_scatter, trajectory_line, current_point]
    
    def update(frame):
        system.update(0.01)
        
        for i, rod in enumerate(system.rods):
            x_data = [rod.point1.x, rod.point2.x]
            y_data = [rod.point1.y, rod.point2.y]
            rods_lines[i].set_data(x_data, y_data)
        
        points_positions = np.array([p.position() for p in system.points])
        points_scatter.set_offsets(points_positions)
        
        if frame > 0:
            trajectory_line.set_data(trajectory_x[:frame], trajectory_y[:frame])
            current_point.set_data([trajectory_x[frame]], [trajectory_y[frame]])
        
        return rods_lines + [points_scatter, trajectory_line, current_point]
    
    ani = FuncAnimation(fig, update, frames=steps, init_func=init, 
                        blit=True, interval=interval, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani