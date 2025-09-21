
import jax
import jax.numpy as jnp
import gymnasium as gym
import sym2real.envs
import hydra
import omegaconf
import numpy as np

from sym2real.util.common import create_one_dim_tr_model
from sym2real.util.replay_buffer import ReplayBuffer
from sym2real.controllers.mpc import MPPIController
from sym2real.controllers.reference_trajectories.setpoint_ref import SetpointReference
from sym2real.controllers.agent import TrajectoryPlanningAgent

import time

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

@hydra.main(config_path="../../sym2real/conf", config_name="main", version_base="1.1")
def run(cfg: omegaconf.DictConfig):

    replay_buffer = ReplayBuffer(capacity=int(1e6),
                                obs_shape=(13,),
                                action_shape=(4,),
                                )
    
    sim_env = gym.make("CF2HoverMujocoEnvAttitudeRate-v0", 
                       cfg=cfg,
                        )
    eval_env = gym.make("CF2HoverMujocoEnvAttitudeRate-v0",
                        cfg=cfg,
                        render_mode="human",
                        )

    dynamics_model = create_one_dim_tr_model(cfg=cfg, 
                                            obs_shape=(13,), 
                                            act_shape=(4,),
                                            model_dir="/home/generalroboticslab/Desktop/crazyflie_controller/01_from_scratch_mppi/CF2HoverMujocoEnvAttitudeRate-v0/hover/env_params_wind_x_0_wind_y_0_wind_z_0_mass_0.027_com_x_0.001/seed_0/symbolic_regression/2025.07.13:142911"
                                            )
    
    # dynamics_model.model.initialize_from_file("/home/generalroboticslab/Desktop/crazyflie_controller/01_from_scratch_mppi/CF2HoverMujocoEnvAttitudeRate-v0/hover/env_params_wind_x_0_wind_y_0_wind_z_0_mass_0.027_com_x_0.001/seed_0/symbolic_regression/2025.07.13:142911/dynamics_model/curr/equations.pkl")
    
    controller = MPPIController(env=sim_env,
                          horizon=40,
                          dt=sim_env.unwrapped.dt,
                          num_samples=4096, 
                          model=dynamics_model,)
    
    reference_class = SetpointReference()
    reference_class.set_center(cfg.overrides.hover_position)
    
    agent = TrajectoryPlanningAgent(controller,
                                    reference_class=reference_class)
    agent.reset()
    obs, _ = sim_env.reset()
    agent.act(t=0, obs=np.array([0, 0, 0,
                                1, 0, 0, 0,
                                0, 0, 0,
                                0, 0, 0])) # warm start

    for i in range(5):
        terminated, truncated = False, False
        obs, _ = eval_env.reset()
        eval_env.render()

        t = 0   
        episode_reward = 0
        
        step_ct = 0
        while not (terminated or truncated):
            action, _ = agent.act(t=t, obs=obs)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            step_ct += 1
            obs = next_obs
            episode_reward += reward
            eval_env.render()
            t+= 1
            
        print("Episode reward:", episode_reward)
        print("Episode length:", t)

run()