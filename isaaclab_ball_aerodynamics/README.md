
# Aerodynamic Models

Math and usage for the batched aerodynamic models used by the adapter in `aerodynamics.py`.

## Model

Let \( \mathbf{v}\in\mathbb{R}^3 \) be world linear velocity, \( \boldsymbol{\omega}\in\mathbb{R}^3 \) world angular velocity, \( \rho \) air density, \( A=\pi r^2 \) area, \( r \) radius, \( C_d \) drag coefficient, \( \kappa \) Magnus factor.

\[
\mathbf{F}_\text{drag} = -\tfrac{1}{2}\rho A C_d \|\mathbf{v}\| \mathbf{v}, \quad
\mathbf{F}_\text{magnus} = \tfrac{1}{2}\rho A r \kappa (\boldsymbol{\omega}\times \mathbf{v})
\]
\[
\mathbf{F}_\text{total}=\mathbf{F}_\text{drag}+\mathbf{F}_\text{magnus},\quad \boldsymbol{\tau}=\mathbf{0}
\]

Shapes: `(N, 3)` for forces and torques. World frame.

## Adapter usage

```python
from isaaclab_ball_aerodynamics.aerodynamics import AeroForceField

aero = AeroForceField(
    device="cuda:0",
    radius_m=0.020,
    air_density=1.225,
    drag_coeff=0.47,
    magnus_factor=0.0
)

# Called once per substep before staging to the sim
aero.apply_to_rigid_object(ball_asset)   # ball_asset: isaaclab.assets.RigidObject
F_w = aero.last_forces_w    # (N, 3)
T_w = aero.last_torques_w   # (N, 3), zero in this model
```

## Where to place calls in an env

* Initialize the adapter **after** the scene is created and assets are resolved.
* Initialize **before** buffer allocation and any `reset(...)` calls.
* In the step loop, call `apply_to_rigid_object(...)` **before** `scene.write_data_to_sim()` on each physics substep.


## Quick example: defining `self.ball`


```python
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)

scene = InteractiveScene(scene_cfg)            # scene_cfg includes ball_cfg
self.scene = scene
self.ball: RigidObject = self.scene["ball"]    # RigidObject handle used by the aero adapter

```


## Direct model API


```python
import torch
from isaaclab_ball_aerodynamics.aero_model import MeasuredAeroModel

model = MeasuredAeroModel(device="cuda:0")
v = torch.randn(1024, 3, device="cuda")
w = torch.randn(1024, 3, device="cuda")
ball_props = {"radii": torch.full((1024,), 0.02, device="cuda"),
              "areas": torch.full((1024,), 3.1415926*0.02**2, device="cuda")}
env_props = {"air_density": 1.225}
params = {"drag_coefficients": torch.full((1024,), 0.43, device="cuda"),
          "magnus_factors":    torch.full((1024,), 0.0,  device="cuda")}
F, T = model.calculate_forces_and_torques(v, w, ball_props, env_props, params)
```


## Tuning

* `drag_coeff`: fit from deceleration vs. speed; typical spheres \~0.4–0.5.
* `magnus_factor`: fit from lateral deflection vs. spin; `0.0` disables Magnus.
* Use SI units and world-frame velocities.