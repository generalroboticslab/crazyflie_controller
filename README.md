# Real-World Attitude Rate Crazyflie 2.1 Controller 

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-purple.svg)]()

</div>

[Easop Lee](https://easoplee.github.io/),
[Samuel A. Moore](https://samavmoore.github.io/), and
[Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>


> Offboard attitude rate control on Crazyflie 2.1 w/ vicon mocap data. 

---

⚠️ **Important Notice**  
This repository **must be used together** with the [Sym2Real](https://github.com/easoplee/sym2real) framework. Please check out this repository first.

- The **Sym2Real repo** provides modular dynamics models, controllers, and simulation environments.  
- This **Crazyflie controller repo** provides the hardware interface for the quadrotor real-world experiments.  

Please **keep both repositories in sync** when running experiments. Updates or mismatches between the two may cause unexpected errors.

---

## 🔗 Syncing with Sym2Real

Make sure you clone the [Sym2Real](https://github.com/easoplee/sym2real) repo, and follow its installation instructions first.

Then, clone this repository **next to** the [Sym2Real](https://github.com/easoplee/sym2real) (e.g. if the parent folder is Desktop):

```
Desktop/
├── sym2real
└── crazyflie-controller
```

```
git clone git@github.com:easoplee/crazyflie-controller.git
```

## 🔧 Building and flashing firmware

Flash the firmware to the Crazyflie hardware.

```
cd crazyflie-firmware
git submodule init
git submodule update --recursive

make cf2_defconfig
make

cfloader flash build/cf2.bin stm32-fw -w radio://0/70/2M/E7E7E7E701 # change the radio address accordingly
```

## 🛠️ Installation

Install additional packages to the previously created sym2real conda environment ([Sym2Real](https://github.com/easoplee/sym2real)).

```
conda activate sym2real

pip install cfclient==2024.7.1

cd crazyflie-lib-python
pip install -e .
```

## ✅ Sanity Check

First, perform this sanity check test to make sure all the components are set up correctly. 
Make sure you have enough space since crazylfie willl start moving.

```
cd tests
python simple_sanity_check.py  # to check cflib: expected to move its propellers 
```

Then, start the vicon streaming. We assume the name is "cf1" here.

```
self.vicon = ViconReader(topic="/vicon/cf1/cf1")
```
The folowing script should print all the states (both from vicon and cflib) and send commands to the crazyflie. Crazyflie should move its propellers without flying away.
```
python readsend_sanity_check.py # check reading and sending commands, tests the whole pipeline
```

## 🚀 Quickstart

Make sure you sync the saved model folders from the experiments in [Sym2Real](https://github.com/easoplee/sym2real). Change the parent folder in `setup.sh`. We assume the following structure:
```
Desktop/
├── sym2real
└── crazyflie_controller
```

```
cd crazyflie_controller
source setup.sh
```

▶️ Run Base Model - Zero-shot (e.g. Symbolic Regression on Crazyflie)
```
python main.py \
    dynamics_model="symbolic_regression" \
    model_path="{path_to_sr_base_model_experiment}"
```
```
e.g. model_path="/home/generalroboticslab/Desktop/sym2real/example_results/CF2HoverMujocoEnvAttitudeRate-v0/hover/env_params_wind_x_0_wind_y_0_wind_z_0_mass_0.027_com_x_0.00/seed_0/symbolic_regression/2025.07.13:142911"
```

These prompts will show up to ask if you want to change the name of the folder or whether you want to collect multiple trajectories.

```
1. Do you want to create a new set of experiments? Then quit and fix the config file. [y/n]: 

# type "n" unless you want to change the saved folder path into config file

2. Please enter the trial number:

# type the trial number (e.g. for zero-shot, type 1. After a model update, for re-deployment type 2, ...)

3. Collect another trajectory? [y/n]:

# if you want to collect another trajectory with this model, hit "y".

4. Enter traj number (reset drone before entering this)

# which trajectory is this. Type an integer.
```

Then, to train the residual model on the collected real world data: 
```
python train_model.py \
    dynamics_model="residual_mlp" \
    dynamics_model.model_a_path="{path_to_sr_base_model_experiment}"
```

```
e.g. dynamics_model.model_a_path="/home/generalroboticslab/Desktop/sym2real/example_results/CF2HoverMujocoEnvAttitudeRate-v0/hover/env_params_wind_x_0_wind_y_0_wind_z_0_mass_0.027_com_x_0.00/seed_0/symbolic_regression/2025.07.13:142911"
```

🪄 Run Residual Model - Finetuned (e.g. Symbolic Regression + Residual MLP on Crazyflie)
```
python main.py \
    dynamics_model="residual_mlp" \
    dynamics_model.model_a="symbolic_regression" \
    dynamics_model.model_a_path="{path_to_sr_base_model_experiment}" \
    model_path="{path_to_residual_mlp_model_experiment}""
```

```
# one folder above the saved .pkl path
e.g. model_path="/home/generalroboticslab/Desktop/crazyflie_controller/real_crazyflie_example_results/2pennies/CF2HoverMujocoEnvAttitudeRate-v0/seed_0/symbolic_regression_residual_mlp/dynamics_model/mlp_model_w_dataset_size_400"
```