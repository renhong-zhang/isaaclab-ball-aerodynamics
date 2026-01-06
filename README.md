
# Isaac Lab Ball Aerodynamics

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Demo Video](https://img.youtube.com/vi/uWAf_B4WidY/0.jpg)](https://www.youtube.com/watch?v=uWAf_B4WidY)

> Click the image above to watch the simulation in action.

GPU batched aerodynamic drag and Magnus forces for spinning balls in Isaac Lab. Implemented in PyTorch. Drop-in force field for `RigidObject` assets.

## Install
```bash
pip install -e .
```


## Quickstart

```python
from isaaclab_ball_aerodynamics.aerodynamics import AeroForceField
from isaaclab.assets.rigid_object import RigidObject

class DummyEnv:
    def __init__(self, sim, scene, ball: RigidObject):
        self.sim = sim
        self.scene = scene
        self.ball = ball

        # Initialize aerodynamics AFTER scene is built, BEFORE resets/buffers
        self.aero = AeroForceField(
            device="cuda:0",
            radius_m=0.020,
            air_density=1.225,
            drag_coeff=0.43,   # set 0.0 to disable
            magnus_factor=0.0
        )

    def step(self):
        # Call once per physics substep, BEFORE scene.write_data_to_sim()
        self.aero.apply_to_rigid_object(self.ball)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=0.01)
```

### Example Ball Definition

The ball used in above example can be defined as follows:
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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.2)),
    ),
)

scene = InteractiveScene(scene_cfg)     # scene_cfg includes ball_cfg
ball: RigidObject = scene["ball"]       # handle used by AeroForceField
```


## Features

* Drag and Magnus forces in world frame
* GPU batched PyTorch compute
* Works with `RigidObject` or collections
* Pluggable model API for custom aero
