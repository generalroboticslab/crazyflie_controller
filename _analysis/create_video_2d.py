import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# Load data
circle_path = "/home/generalroboticslab/Desktop/crazyflie_controller/real_cf_test_07.10_exp_results/seed=1/symbolic_regression_residual_mlp/real_data/trial_3/traj_2.csv"
circle_data = pd.read_csv(circle_path)

xyz_real_pos = circle_data.iloc[:, 1:4].to_numpy()
xyz_des_pos = circle_data.iloc[:, 18:21].to_numpy()
assert xyz_real_pos.shape == xyz_des_pos.shape

# Setup 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits based on all data
all_xyz = np.concatenate([xyz_real_pos, xyz_des_pos], axis=0)
ax.set_xlim(np.min(all_xyz[:, 0]), np.max(all_xyz[:, 0]))
ax.set_ylim(np.min(all_xyz[:, 1]), np.max(all_xyz[:, 1]))
ax.set_zlim(np.min(all_xyz[:, 2]), np.max(all_xyz[:, 2]))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Real vs Desired Trajectory')

# Initialize plot elements
real_line, = ax.plot([], [], [], lw=2, color='blue', label='Real')
des_line, = ax.plot([], [], [], lw=2, color='green', linestyle='--', label='Desired')
real_point, = ax.plot([], [], [], 'bo')
des_point, = ax.plot([], [], [], 'go')
ax.legend()

# Init function
def init():
    real_line.set_data([], [])
    real_line.set_3d_properties([])
    
    des_line.set_data([], [])
    des_line.set_3d_properties([])
    
    real_point.set_data([], [])
    real_point.set_3d_properties([])
    
    des_point.set_data([], [])
    des_point.set_3d_properties([])
    
    return real_line, des_line, real_point, des_point

# Update function
def update(frame):
    real_line.set_data(xyz_real_pos[:frame, 0], xyz_real_pos[:frame, 1])
    real_line.set_3d_properties(xyz_real_pos[:frame, 2])
    
    des_line.set_data(xyz_des_pos[:frame, 0], xyz_des_pos[:frame, 1])
    des_line.set_3d_properties(xyz_des_pos[:frame, 2])
    
    real_point.set_data([xyz_real_pos[frame, 0]], [xyz_real_pos[frame, 1]])
    real_point.set_3d_properties([xyz_real_pos[frame, 2]])
    
    des_point.set_data([xyz_des_pos[frame, 0]], [xyz_des_pos[frame, 1]])
    des_point.set_3d_properties([xyz_des_pos[frame, 2]])
    
    return real_line, des_line, real_point, des_point

# Create animation
anim = FuncAnimation(
    fig, update, frames=len(xyz_real_pos), init_func=init,
    blit=False, interval=20  # 50 FPS
)


# Save animation
writer = FFMpegWriter(fps=50, metadata=dict(artist='Crazyflie'), bitrate=1800)
anim.save("real_vs_desired_xyz_symbolic_regression_residual_mlp_update2.mp4", writer=writer)
plt.close()
