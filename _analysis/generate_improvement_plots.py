# sr* + mlp improvements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "/home/generalroboticslab/Desktop/crazyflie-controller/real_cf_test_07.22_results_noload/CF2HoverMujocoEnvAttitudeRate-v0/seed_1/symbolic_regression_residual_mlp"


direct_path = f"{path}/real_data/trial_1/traj_1.csv"
first_path = f"{path}/real_data/trial_2/traj_1.csv"
second_path = f"{path}/real_data/trial_3/traj_1.csv"
third_path = f"{path}/real_data/trial_4/traj_1.csv"

direct_df = pd.read_csv(direct_path)
first_df = pd.read_csv(first_path)
second_df = pd.read_csv(second_path)
third_df = pd.read_csv(third_path)

def calculate_mse(df):
    # x y z
    mse = 0
    for i in range(len(df)):
        mse += (df.iloc[i]['pos_x'] - df.iloc[i]['desired_pos_x'])**2 + (df.iloc[i]['pos_y'] - df.iloc[i]['desired_pos_y'])**2 + (df.iloc[i]['pos_z'] - df.iloc[i]['desired_pos_z'])**2
    return mse / len(df)

direct_mse = calculate_mse(direct_df)
first_mse = calculate_mse(first_df)
second_mse = calculate_mse(second_df)
third_mse = calculate_mse(third_df)

mse_plot = [direct_mse, first_mse, second_mse]
# mse_plot_labels = ['direct', 'first', 'second', 'third']
plt.plot(mse_plot, )
plt.legend()
plt.savefig('mse_plot.png')

# compare the direct trajectory with the third trajectory

def create_3d_plot(data, labels, colors):
    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    gt_xyz = data[0].iloc[:, 18:21].to_numpy()
    gt_xyz_shifted = gt_xyz - gt_xyz[0]
    ax3d.plot(gt_xyz_shifted[:, 0], gt_xyz_shifted[:, 1], gt_xyz_shifted[:, 2],
              label="Reference", color="black", alpha=0.6, linestyle='--', linewidth=2)
    
    for i, (df, label) in enumerate(zip(data, labels)):
        xyz = df.iloc[:, 1:4].to_numpy()
        xyz_shifted = xyz - xyz[0]
        ax3d.plot(xyz_shifted[:, 0], xyz_shifted[:, 1], xyz_shifted[:, 2],
              label=f"{label}", color=colors[i], alpha=0.85, linewidth=2)
    
    ax3d.set_title('3D Trajectory Comparison')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('3d_plot.png')

def create_position_error_plot(data, labels, colors):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot cumulative position error for each trajectory
    for i, (df, label) in enumerate(zip(data, labels)):
        # Calculate position error: actual position - reference position
        pos_err = df.iloc[:, 1:4].to_numpy() - df.iloc[:, 18:21].to_numpy()
        pos_error_sum = np.sum(np.abs(pos_err), axis=1)
        cumsum = np.cumsum(pos_error_sum)
        x = np.arange(len(cumsum))
        
        ax.plot(x, cumsum, label=f"{label}", color=colors[i], alpha=0.85, linewidth=2)
        ax.fill_between(x, cumsum, color=colors[i], alpha=0.25)
    
    ax.set_title('Cumulative Position Error')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Position Error (m)')
    ax.legend(loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig('position_error_plot.png')

def create_x_y_z_plot(data, labels, colors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot X error
    for i, (df, label) in enumerate(zip(data, labels)):
        x_error = df.iloc[:, 1].to_numpy() - df.iloc[:, 18].to_numpy()
        axes[0].plot(x_error, label=f"{label}", color=colors[i], alpha=0.85, linewidth=2)
    axes[0].set_title('X Position Error')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('X Error (m)')
    axes[0].legend()
    
    # Plot Y error
    for i, (df, label) in enumerate(zip(data, labels)):
        y_error = df.iloc[:, 2].to_numpy() - df.iloc[:, 19].to_numpy()
        axes[1].plot(y_error, label=f"{label}", color=colors[i], alpha=0.85, linewidth=2)
    axes[1].set_title('Y Position Error')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Y Error (m)')
    axes[1].legend()
    
    # Plot Z error
    for i, (df, label) in enumerate(zip(data, labels)):
        z_error = df.iloc[:, 3].to_numpy() - df.iloc[:, 20].to_numpy()
        axes[2].plot(z_error, label=f"{label}", color=colors[i], alpha=0.85, linewidth=2)
    axes[2].set_title('Z Position Error')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Z Error (m)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('xyz_error_plot.png')

# Generate colors using a colormap similar to plotter.py
files = [direct_df, first_df, second_df]
colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(files)))  # 3 colors for 3 dataframes
create_position_error_plot(files, ['direct transfer', 'update 1', 'update 2'], colors)
create_x_y_z_plot(files, ['direct', 'first', 'second'], colors)






