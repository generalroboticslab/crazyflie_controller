

import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def qdistance(q1, q2):
    dot = jnp.sum(q1 * q2, axis=-1)
    return 1.0 - dot**2

def compute_cost_components(states, state_refs):
    pos_err = states[:, 0:3] - state_refs[:, 0:3]
    quat_err = qdistance(states[:, 3:7], state_refs[:, 3:7])
    lin_vel_err = states[:, 7:10] - state_refs[:, 7:10]
    ang_vel_err = states[:, 10:13] - state_refs[:, 10:13]

    pos_cost = alpha_p * jnp.linalg.norm(pos_err, axis=-1)
    quat_cost = alpha_r * quat_err
    lin_vel_cost = alpha_v * jnp.linalg.norm(lin_vel_err, axis=-1)
    ang_vel_cost = alpha_omega * jnp.linalg.norm(ang_vel_err, axis=-1)

    total_cost = pos_cost + quat_cost + lin_vel_cost + ang_vel_cost
    return total_cost, pos_cost, quat_cost, lin_vel_cost, ang_vel_cost

def calculate_episode_cost(csv_file):
    """Calculate the total cost for a single episode from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        states = df.iloc[:, 1:14].values
        refs = df.iloc[:, 18:31].values
        
        total_cost, pos_cost, quat_cost, lin_vel_cost, ang_vel_cost = compute_cost_components(states, refs)
        
        return np.mean(total_cost)
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def get_model_costs(base_path, model_type, trials=[1, 2, 3]):
    """Calculate costs for a specific model across trials"""
    model_path = os.path.join(base_path, model_type, "real_data")
    costs = []
    
    for trial in trials:
        trial_path = os.path.join(model_path, f"trial_{trial}")
        
        # Look for CSV files in the trial folder
        if os.path.exists(trial_path):
            csv_files = [f for f in os.listdir(trial_path) if f.endswith('.csv')]
            if csv_files:
                # Use the first CSV file (or you can specify which one)
                csv_file = os.path.join(trial_path, csv_files[0])
                cost = calculate_episode_cost(csv_file)
                if cost is not None:
                    costs.append(cost)
                else:
                    print(f"Could not calculate cost for {csv_file}")
            else:
                print(f"No CSV files found in {trial_path}")
        else:
            print(f"Trial path not found: {trial_path}")
    
    return costs

# ---- Config ----
alpha_p = 5.0 
alpha_r = 3.0 
alpha_omega = 0.1 
alpha_v = 0.05 

# ---- Load data ----
base_path = "/home/generalroboticslab/Desktop/crazyflie_controller/real_cf_test_07.13_results/CF2HoverMujocoEnvAttitudeRate-v0/seed_0"

plot_variants = {
    "sym_res_mlp": "symbolic_regression_residual_mlp",
    "mlp_res_mlp": "gaussian_mlp_ensemble_residual_mlp",
    "sym": "symbolic_regression",
    "sym_res_sym": "symbolic_regression_residual_sr",
    "mlp_res_sym": "gaussian_mlp_ensemble_residual_sr",
}

# Define your custom colors
variant_colors = {
    "sym_res_mlp": "#4682B4",   # Steel Blue
    "sym_res_sym": "#D17C5B",   # Light Brown
    "mlp_res_mlp": "#8FBC8F",   # Dark Sea Green
    "mlp_res_sym": "#9E7BAF",   # Plum
    "sym": "#FFD700"            # Gold
}

# Labels for the plot
variant_labels = {
    "sym_res_mlp": "SR* + NN (res)",
    "mlp_res_mlp": "NN* + NN (res)",
    "sym_res_sym": "SR* + SR (res)",
    "mlp_res_sym": "NN* + SR (res)",
    "sym": "SR (fine-tuned)"
}

# Calculate costs for each model
model_results = {}
for variant_key, model_type in plot_variants.items():
    print(f"Processing {variant_key} ({model_type})...")
    costs = get_model_costs(base_path, model_type, trials=[1, 2, 3])
    if costs:
        model_results[variant_key] = costs
        print(f"  Costs: {costs}")
    else:
        print(f"  No valid costs found for {variant_key}")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each model's progression
for variant_key, costs in model_results.items():
    if len(costs) > 0:
        trials = list(range(1, len(costs) + 1))
        color = variant_colors.get(variant_key, "#9E7BAF")
        label = variant_labels.get(variant_key, variant_key)
        
        plt.plot(trials, costs, 'o-', label=label, color=color, linewidth=2.5, markersize=8)

if model_results:
    plt.xlabel("Trial Number")
    plt.ylabel("Episode Cost")
    plt.title("Model Performance Progression Across Trials")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")

    # Set integer ticks for trial numbers
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("model_progression_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as 'model_progression_comparison.png'")
else:
    print("No valid data found to plot!")