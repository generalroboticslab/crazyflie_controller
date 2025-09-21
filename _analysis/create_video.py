import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# File paths
base_path = "/home/generalroboticslab/Desktop/crazyflie_controller/real_cf_test_07.10_exp_results/seed=1/symbolic_regression_residual_mlp/real_data"
file_paths = [
    f"{base_path}/trial_1/traj_2.csv",
    f"{base_path}/trial_2/traj_2.csv",
    f"{base_path}/trial_3/traj_2.csv",
]

# Load data
real_trajs = []
des_trajs = []

for path in file_paths:
    data = pd.read_csv(path)
    real_xy = data.iloc[:, 1:3].to_numpy()
    des_xy = data.iloc[:, 18:20].to_numpy()

    # Normalize so the first desired point is at origin
    offset = des_xy[0]
    real_xy -= offset
    des_xy -= offset

    real_trajs.append(real_xy)
    des_trajs.append(des_xy)

# Determine axis limits
all_xy = np.concatenate(real_trajs + [des_trajs[0]], axis=0)  # Only first desired trajectory
x_min, x_max = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
y_min, y_max = np.min(all_xy[:, 1]), np.max(all_xy[:, 1])

# Setup figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Overlayed 2D XY Real Trajectories with One Reference')

colors = ['blue', 'red', 'purple']
markers = ['bo', 'ro', 'mo']

# Initialize plot elements
real_lines = []
real_points = []

for i in range(len(real_trajs)):
    real_line, = ax.plot([], [], lw=2, color=colors[i], linestyle='-', label=f'Real {i+1}')
    real_point, = ax.plot([], [], markers[i])
    real_lines.append(real_line)
    real_points.append(real_point)

# Only one reference (desired) trajectory
des_line, = ax.plot([], [], lw=2, color='black', linestyle='--', label='Reference')
des_point, = ax.plot([], [], 'ko')

ax.legend()

# Init function
def init():
    for rl, rp in zip(real_lines, real_points):
        rl.set_data([], [])
        rp.set_data([], [])
    des_line.set_data([], [])
    des_point.set_data([], [])
    return real_lines + [des_line] + real_points + [des_point]

# Update function
def update(frame):
    for i in range(len(real_trajs)):
        if frame < len(real_trajs[i]):
            real_lines[i].set_data(real_trajs[i][:frame, 0], real_trajs[i][:frame, 1])
            real_points[i].set_data([real_trajs[i][frame, 0]], [real_trajs[i][frame, 1]])
    if frame < len(des_trajs[0]):
        des_line.set_data(des_trajs[0][:frame, 0], des_trajs[0][:frame, 1])
        des_point.set_data([des_trajs[0][frame, 0]], [des_trajs[0][frame, 1]])
    return real_lines + [des_line] + real_points + [des_point]

# Determine maximum length among all trajectories
max_frames = max(len(x) for x in real_trajs)

# Create animation
anim = FuncAnimation(
    fig, update, frames=max_frames, init_func=init,
    blit=True, interval=20
)

# Save animation
writer = FFMpegWriter(fps=50, metadata=dict(artist='Crazyflie'), bitrate=1800)
anim.save("overlayed_real_vs_desired_xy_one_reference.mp4", writer=writer)
plt.close()
