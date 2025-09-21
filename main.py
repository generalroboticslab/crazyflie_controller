from email import parser
import dynamics_model
import rospy
import cflib.crtp
from src.vicon_reader import ViconReader
from src.crazyflie_interface import CrazyflieInterface
import numpy as np
import signal
import time

from src.pd_controller import AttitudeStabilizer

from scipy.spatial.transform import Rotation as R

from sym2real.util.common import create_one_dim_tr_model
from sym2real.controllers.agent import TrajectoryPlanningAgent
from sym2real.controllers.reference_trajectories.setpoint_ref import SetpointReference
from sym2real.controllers.reference_trajectories.cf_circle_ref import CrazyflieCircleReference
from sym2real.controllers.reference_trajectories.cf_lemniscate_ref import CrazyflieLemniscateReference
from sym2real.controllers.mpc.mppi import MPPIController

from src.logger import Logger

import hydra
import gymnasium as gym
import omegaconf
import os
import sym2real.envs

import pyfiglet
import jax
import sys
import jax.numpy as jnp

import argparse

CTRL_HZ = 50.0

PD_TOTAL_STEPS = 300
MPPI_TOTAL_STEPS = 700 
PD_LAND_TOTAL_STEPS = 1000 

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully to avoid deadlocks."""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    
    try:
        jax.clear_caches()
        jax.device_get(jax.numpy.array([1.0]))
        print("✓ JAX GPU resources cleaned up")
    except Exception as e:
        print(f"⚠ JAX cleanup warning: {e}")
    
    # Force cleanup of Julia resources
    try:
        import pysr
        # PySR doesn't have explicit cleanup, but we can try to force garbage collection
        import gc
        gc.collect()
        print("✓ Julia resources cleaned up")
    except Exception as e:
        print(f"⚠ Julia cleanup warning: {e}")
    
    print("Cleanup complete. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def clean_env_params(env_params):
    """Convert env_params dict to clean string without quotes/spaces"""
    return "_".join(f"{k}_{v}" for k, v in env_params.items())

def get_dynamics_model_dir(dynamics_model_choice, dynamics_model_config):
    """Generate appropriate directory name for dynamics model, handling residual models specially"""
    if dynamics_model_choice.startswith("residual"):
        try:
            if isinstance(dynamics_model_config, str):
                if 'model_a' in dynamics_model_config:
                    import re
                    match = re.search(r"'model_a':\s*'([^']*)'", dynamics_model_config)
                    if match:
                        model_a_value = match.group(1)
                        return f"{model_a_value}_{dynamics_model_choice}"
            else:
                if hasattr(dynamics_model_config, 'model_a') and dynamics_model_config.model_a:
                    return f"{dynamics_model_config.model_a}_{dynamics_model_choice}"
        except:
            pass
        return dynamics_model_choice
    else:
        return dynamics_model_choice

# Register the custom resolvers
omegaconf.OmegaConf.register_new_resolver("clean_env_params", clean_env_params)
omegaconf.OmegaConf.register_new_resolver("get_dynamics_model_dir", get_dynamics_model_dir)

class CrazyflieControlNode:
    def __init__(self, cfg, sim_env, which_trial, traj_num, enable_logging=True):
        self.cfg = cfg
        self.sim_env = sim_env
        self.enable_logging = enable_logging
        
        self.max = 0
        work_dir = os.getcwd()

        if self.enable_logging:
            folder = f"{work_dir}/real_data/"
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            trial_folder = f"{folder}/trial_{which_trial}"
            if not os.path.exists(trial_folder):
                os.makedirs(trial_folder)
            filename = f"{trial_folder}/traj_{traj_num}.csv"
            self.logger = Logger(filename=filename)
            
        # ----------------------------------------------------------------------
        #  DYNAMICS MODEL
        # ----------------------------------------------------------------------
        self.dynamics_model = create_one_dim_tr_model(cfg=cfg, 
                                                      obs_shape=sim_env.observation_space.shape, 
                                                      act_shape=sim_env.action_space.shape,
                                                      model_dir=cfg.model_path,
                                                    )
        
        # ----------------------------------------------------------------------
        #  CONTROL AGENT (CEM)
        # ----------------------------------------------------------------------
        if cfg.overrides.reference_type == "circle":
            self.reference_class = CrazyflieCircleReference()
        elif cfg.overrides.reference_type == "lemniscate":
            self.reference_class = CrazyflieLemniscateReference()
        elif cfg.overrides.reference_type == "hover":
            self.reference_class = SetpointReference()
        
        
        if "CF2" in cfg.overrides.env:
            rollout_horizon_in_sec = 0.8
        elif "Mushr" in cfg.overrides.env:
            rollout_horizon_in_sec = 0.5
            
        num_samples = 1024
        self.controller = MPPIController(
            env=sim_env, 
            horizon=int(rollout_horizon_in_sec/sim_env.unwrapped.dt),
            dt=0.01,
            num_samples=num_samples, 
            model=self.dynamics_model,
            seed=cfg.seed
        )

        self.agent = TrajectoryPlanningAgent(self.controller, reference_class=self.reference_class)

        self.shutdown_flag = False
        signal.signal(signal.SIGINT, self.signal_handler)
        
        cflib.crtp.init_drivers()
        rospy.init_node("cf_vicon_ctrl", anonymous=True)
        self.rate = rospy.Rate(CTRL_HZ)
        
        self.vicon = ViconReader()
        self.cf = CrazyflieInterface()
        self.hover_controller = AttitudeStabilizer()

        time.sleep(0.5) # Allow time for Crazyflie to initialize
        init_pos, _ = self.vicon.get_state()
        self.init_pos = np.array(init_pos)
        self.init_pos[2] = 0.5
        
        setpoint = self.init_pos[:3]  # Goal position (x, y, z)
        self.reference_class.set_center(setpoint)
        
        self.step_ct = 0
        
        # Warm-up CEM model
        for i in range(3):
            self.agent.act(t=0, obs=np.zeros((13,)))
        
    
    def signal_handler(self, sig, frame):
        self.shutdown_flag = True

    def in_boundary(self, pos):
        x, y, z = pos
        return not (abs(x) > 1.5 or abs(y) > 1.5 or z > 1.7)

    def acc_to_pwm(self, acc_des_z):
        pwm = int(acc_des_z * 132000 * self.sim_env.unwrapped.mass)
        return np.clip(pwm, 0, 60000)
    
    def run(self):
        print("shutdown flag: ", self.shutdown_flag)
        self.cf.set_led_off()
        while not rospy.is_shutdown() and not self.shutdown_flag and self.step_ct <= PD_LAND_TOTAL_STEPS:
            t_start = time.time()
            pos, vel = self.vicon.get_state()
            if not self.in_boundary(pos):
                print("out of boundary; put the damn thing in the workspace")
                break

            euler = self.cf.get_euler_degs()
            if np.abs(euler[0]) > 70 or np.abs(euler[1]) > 70: # dangerous attitude
                break
            rate_readings = self.cf.get_omega_degs()
            
            omega = np.radians(rate_readings)
            # convert euler to quaternion
            quat_xyzw = R.from_euler('xyz', euler, degrees=True).as_quat()
            quat_wxyz = np.roll(quat_xyzw, 1)

            print(f"Position: {pos}, Velocity: {vel}, Euler: {euler}, rate_readings: {rate_readings}")

            if pos is None or euler is None:
                self.rate.sleep()
                continue

            if self.step_ct < PD_TOTAL_STEPS:
                angvel, thrust = self.hover_controller.control(euler, rate_readings, pos, vel)
                action = np.hstack((thrust, angvel))  # thrust and angular velocity
            elif self.step_ct <= MPPI_TOTAL_STEPS:
                if self.step_ct == PD_TOTAL_STEPS:
                    self.reference_class.set_center(np.array([pos[0], pos[1], pos[2]]))
                    print(pyfiglet.figlet_format("INFO: MPPI Started!"))
                    self.cf.set_led_orange()
                    # self.cf.set_led_blue()

                action, _ = self.agent.act(t=self.step_ct-PD_TOTAL_STEPS, obs=np.hstack((pos, quat_wxyz, vel, omega)))
                angvel = action[1:]

                angvel = np.degrees(angvel)  
                thrust = self.acc_to_pwm(action[0]/self.sim_env.unwrapped.mass)
                
                if self.enable_logging:
                    self.logger.log(time.time(), 
                                    np.hstack((pos, quat_wxyz, vel, omega)), 
                                    angvel,
                                    action[0], # this is in newtons
                                    self.reference_class.get_ref_state(self.step_ct-PD_TOTAL_STEPS),
                                    )
            elif self.step_ct <= PD_LAND_TOTAL_STEPS:
                
                if self.step_ct == MPPI_TOTAL_STEPS + 1:
                    print("landing start")
                    self.cf.set_led_off()
                angvel, thrust = self.hover_controller.control_landing(euler, rate_readings, pos, vel)
                action = np.hstack((thrust, angvel))  # thrust and angular velocity
                
                if pos[2] < 0.07:
                    print("landing complete")
                    break

            self.cf.send_control(angvel, thrust)

            t_elapsed = time.time() - t_start
            if t_elapsed - 1.0/CTRL_HZ > 0:
                print(pyfiglet.figlet_format(f"Warning: Control loop is taking too long at {self.step_ct}! {t_elapsed - 1.0/CTRL_HZ:.3f} seconds over budget."))
            self.step_ct += 1
            
            self.rate.sleep()

        self.cf.stop()
        print("Shutdown complete.")
        
        if self.enable_logging:
            self.logger.close()
            print("Logging complete.")
            
        print("Max MPPI time: ", self.max)
    
if __name__ == "__main__":

    @hydra.main(config_path="../sym2real/conf", config_name="main_real_crazyflie", version_base="1.1")
    def run(cfg: omegaconf.DictConfig):
        
        safety_check = input("Do you want to create a new set of experiments? Then quit and fix the config file. [y/n]: ").strip().lower()
        if safety_check == "y":
            print("Quitting... Please change experiment name in the config file.") # quit
            return
        elif safety_check == "n":
            which_trial = input("Please enter the trial number: ")
            try:
                which_trial = int(which_trial)
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
        
        while True:
            cont = input("Collect another trajectory? [y/n]: ").strip().lower()
            if cont != "y":
                break
            
            traj_num = int(input("Enter traj number (reset drone before entering this): "))
            
            if cfg.overrides.env == "CF2HoverMujocoEnvAttitudeRate-v0":
                pass 
            else:
                # throw an error
                raise ValueError("Invalid environment name. Please use 'CF2HoverMujocoEnvAttitudeRate-v0'.")

            sim_env = gym.make(cfg.overrides.env, 
                            cfg=cfg,
                        )
            
            node = CrazyflieControlNode(cfg,
                                        sim_env,
                                        which_trial,
                                        traj_num,
                                        enable_logging=True)
            node.run()

    run()