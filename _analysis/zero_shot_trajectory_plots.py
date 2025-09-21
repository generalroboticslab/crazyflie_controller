mlp_csv_path = "/home/generalroboticslab/Desktop/crazyflie-controller/.real_exp_results/real_cf_test_0707_exp_results/hover/seed=1/real_data/trial_1/traj_1.csv"
sr_csv_path = "/home/generalroboticslab/Desktop/crazyflie-controller/.real_exp_results/real_cf_test_0707_exp_results/hover/seed=1/real_data/trial_1/traj_2.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# Load MLP and SR trajectories
mlp_data = pd.read_csv(mlp_csv_path)
sr_data = pd.read_csv(sr_csv_path)

# Extract real and desired positions (X,Y,Z)
mlp_real = mlp_data.iloc[:, 1:4].to_numpy()
mlp_des  = mlp_data.iloc[:, 18:21].to_numpy()
sr_real  = sr_data.iloc[:, 1:4].to_numpy()
sr_des   = sr_data.iloc[:, 18:21].to_numpy()

assert mlp_real.shape == mlp_des.shape == sr_real.shape == sr_des.shape
T = min(len(mlp_real), len(sr_real))

# Set global XYZ limits
all_xyz = np.vstack([mlp_real, mlp_des, sr_real, sr_des])
xyz_min = np.min(all_xyz, axis=0)
xyz_max = np.max(all_xyz, axis=0)

# Setup figure and subplots
fig = plt.figure(figsize=(12, 6))
ax_mlp = fig.add_subplot(121, projection='3d')
ax_sr = fig.add_subplot(122, projection='3d')

for ax, title in zip([ax_mlp, ax_sr], ["MLP Direct Transfer", "SR Direct Transfer"]):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)

# MLP elements
mlp_real_line, = ax_mlp.plot([], [], [], lw=2, color='blue', label='Real')
mlp_des_line,  = ax_mlp.plot([], [], [], lw=2, color='green', linestyle='--', label='Desired')
mlp_real_dot,  = ax_mlp.plot([], [], [], 'bo')
mlp_des_dot,   = ax_mlp.plot([], [], [], 'go')

# SR elements
sr_real_line, = ax_sr.plot([], [], [], lw=2, color='blue', label='Real')
sr_des_line,  = ax_sr.plot([], [], [], lw=2, color='green', linestyle='--', label='Desired')
sr_real_dot,  = ax_sr.plot([], [], [], 'bo')
sr_des_dot,   = ax_sr.plot([], [], [], 'go')

# Legends
ax_mlp.legend()
ax_sr.legend()

# Init function
def init():
    for line in [mlp_real_line, mlp_des_line, sr_real_line, sr_des_line]:
        line.set_data([], [])
        line.set_3d_properties([])
    for dot in [mlp_real_dot, mlp_des_dot, sr_real_dot, sr_des_dot]:
        dot.set_data([], [])
        dot.set_3d_properties([])
    return mlp_real_line, mlp_des_line, mlp_real_dot, mlp_des_dot, sr_real_line, sr_des_line, sr_real_dot, sr_des_dot

# Update function
def update(i):
    # MLP
    mlp_real_line.set_data(mlp_real[:i, 0], mlp_real[:i, 1])
    mlp_real_line.set_3d_properties(mlp_real[:i, 2])
    mlp_real_dot.set_data([mlp_real[i, 0]], [mlp_real[i, 1]])
    mlp_real_dot.set_3d_properties([mlp_real[i, 2]])

    mlp_des_line.set_data(mlp_des[:i, 0], mlp_des[:i, 1])
    mlp_des_line.set_3d_properties(mlp_des[:i, 2])
    mlp_des_dot.set_data([mlp_des[i, 0]], [mlp_des[i, 1]])
    mlp_des_dot.set_3d_properties([mlp_des[i, 2]])

    # SR
    sr_real_line.set_data(sr_real[:i, 0], sr_real[:i, 1])
    sr_real_line.set_3d_properties(sr_real[:i, 2])
    sr_real_dot.set_data([sr_real[i, 0]], [sr_real[i, 1]])
    sr_real_dot.set_3d_properties([sr_real[i, 2]])

    sr_des_line.set_data(sr_des[:i, 0], sr_des[:i, 1])
    sr_des_line.set_3d_properties(sr_des[:i, 2])
    sr_des_dot.set_data([sr_des[i, 0]], [sr_des[i, 1]])
    sr_des_dot.set_3d_properties([sr_des[i, 2]])

    return mlp_real_line, mlp_des_line, mlp_real_dot, mlp_des_dot, sr_real_line, sr_des_line, sr_real_dot, sr_des_dot

# Create animation
anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=20, blit=False)

# Save
writer = FFMpegWriter(fps=50, metadata=dict(artist='Crazyflie'), bitrate=2000)
anim.save("/home/generalroboticslab/Desktop/crazyflie-controller/_analysis/cf_3d_mlp_vs_sr.mp4", writer=writer)
print("✅ Animation saved.")
