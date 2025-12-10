# 🦿 Model Predictive Control for Bipedal Locomotion

This repository implements a Model Predictive Control (MPC) framework for bipedal walking based on the Linear Inverted Pendulum Model (LIPM). It includes

- kinematic footstep generation  
- ZMP regulation with rotated support polygon constraints  
- external force perturbations
- nonlinear trajectory extension for curved walking  
- optional centrifugal force  
- 2D plots, top view visualization and 3D humanoid animation with Pinocchio and Meshcat  

The code is intended for humanoid robotics research and walking stability analysis.

## 📁 Project structure

```text
.
├── articles/                # Scientific papers used as references
├── main3.py                 # Main MPC simulation with perturbations
├── ZMP.py                   # 1D and 2D MPC controllers
├── steps_generator.py       # Kinematic footstep generator
├── vizualisation.py         # 2D plots, top view and 3D visualization tools
├── environment.yml          # Conda environment definition
└── ROBOT projet.png         # Project poster

```  

- `main3.py` runs the full simulation loop, applies external forces, detects falls and calls the MPC solver  

- `ZMP.py` contains the LIPM based MPC controllers in one and two dimensions  

- `steps_generator.py` generates a complete sequence of footsteps from a commanded velocity  

- `vizualisation.py` provides plotting utilities and humanoid visualization with inverse kinematics  


## ⚙️ Installation

Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate <env_name>

```

Main dependencies

- numpy  
- matplotlib  
- qpsolvers (with OSQP)  
- tqdm  
- pinocchio  
- meshcat  
- example_robot_data  

## ▶️ Running the simulation

From the root of the repository, run

```bash
python main3.py
```

This will

- generate footsteps from the commanded velocity  
- solve the 2D MPC problem at each time step  
- optionally apply external forces and centrifugal effects  
- display plots and visualizations automatically  

If you want to enable the humanoid animation with Pinocchio and Meshcat, uncomment the corresponding call in `main3.py`

```python
vizualisation.run_robot_visualization(history_ref, indicator, history_com, history_zmp,
                                      foot_size_lat, foot_size_fwd, full_teta, T_step, v_max)

```

## 🧠 Dynamic model and MPC formulation

The walking controller is based on the Linear Inverted Pendulum Model with state

$$
x = [c,\ \dot c,\ \ddot c]
$$

and ZMP output

$$
z = c - \frac{h}{g}\,\ddot c
$$

The MPC uses jerk as control input, tracks a reference ZMP trajectory and enforces ZMP constraints inside the support polygon of the stance foot. The support region is modeled as a rectangle that can be rotated according to the foot orientation.

The two dimensional formulation stacks the sagittal and lateral directions into a single optimization problem, which allows the controller to consider coupled X and Y motions.


## 👣 Footstep generation

The file `steps_generator.py` implements a kinematic template based footstep generator

- step duration adapts to the norm of the commanded linear velocity  
- a simple integration of the body motion provides a nominal next foot pose  
- sagittal and lateral constraints keep steps within kinematic bounds  
- left and right feet alternate automatically  

The output is a list of footsteps, each with position, orientation, duration and support side.

## 🌀 Nonlinear trajectory extension

Beyond straight line walking with decoupled X and Y motions, the project also supports a nonlinear walking trajectory extension

- the reference motion in the horizontal plane can follow curved paths  
- X and Y motions are treated together to generate turning or bent trajectories  
- ZMP constraints, kinematic bounds and stability conditions are still enforced  
- the resulting footsteps and CoM motion allow the robot to walk along non straight paths while remaining dynamically stable  

This makes it possible to study more realistic humanoid gaits such as turning, gentle curves and complex trajectories.

## 💥 External and centrifugal forces

In `main3.py` you can configure

- a lateral or forward external force applied during a chosen time interval  
- an optional centrifugal effect derived from the commanded angular velocity and the instantaneous CoM velocity  

These are converted into additional accelerations that act on the CoM dynamics. The effect on ZMP and stability can be visualized directly in the plots.


## 📊 Visualization

The project provides several visualization modes

### 2D lateral view

- lateral CoM position over time  
- lateral ZMP trajectory  
- time varying lateral ZMP bounds associated with the stance foot  

### Top view (X–Y plane)

- full CoM trajectory  
- full ZMP trajectory  
- rotated foot rectangles for each step  
- color code for left and right footsteps  
- start and end markers for the CoM path  

### 3D humanoid animation

Using Pinocchio, Meshcat and the Talos model, the project can animate

- the humanoid configuration following the planned CoM trajectory via inverse kinematics  
- the motion of both feet  
- a marker representing the ZMP and the support polygon  

This is useful to qualitatively assess gait realism and robustness.

## 🧪 Fall detection

A simple fall detection logic is included

- at each time step, the ZMP is checked against the current support polygon  
- a counter increases while the ZMP is outside and resets otherwise  
- if the counter exceeds a threshold, a fall is declared and the corresponding time is printed  

This allows quick evaluation of the controller robustness under strong perturbations.


## 🎯 Goals

The main goals of this repository are

- implement a complete MPC based walking controller for a bipedal robot  
- study stability with explicit constraints on ZMP and terminal behavior  
- analyze robustness with respect to external perturbations and centrifugal effects  
- demonstrate an extension to nonlinear trajectories that includes turning and curved motion  
- provide clear visual tools for understanding the generated gaits  


## 🧑‍💻 Author

This code base was developed by Swann Cordier and Alexandre Mallez as part of our academic project during the MVA Robotics class. The associated presentation poster is available as `ROBOT projet.png` in this repository.

## 📚 Scientific references

The `articles` folder contains the main scientific papers that supported the development of this project. These references cover

- Linear Inverted Pendulum Model for bipedal walking  
- ZMP based walking control  
- Model Predictive Control for humanoid locomotion  
- Nonlinear trajectory generation for walking and turning motions  

They were used both for the theoretical formulation of the MPC and for the implementation of the walking pattern generator and stability constraints.

