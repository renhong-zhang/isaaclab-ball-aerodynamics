
# Isaac Lab Ball Aerodynamics

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.21690-b31b1b.svg)](https://arxiv.org/abs/2509.21690)

<div align="center">
  <a href="https://www.youtube.com/watch?v=uWAf_B4WidY">
    <img src="https://img.youtube.com/vi/uWAf_B4WidY/0.jpg" width="50%">
  </a>

  <p style="margin-top: 8px; text-align: center;">
    <em>Click the image above to watch the simulation in action</em>
  </p>
</div>


### **GPU-batched Aerodynamic Drag and Magnus Forces for Spinning Balls in Isaac Lab.**

This repository serves as the **official implementation** of the aerodynamics simulation backbone used in our research on **Humanoid Table Tennis** ([arXiv:2509.21690](https://arxiv.org/abs/2509.21690)). It provides a drop-in, physically accurate force field for `RigidObject` assets in **Isaac Lab**, enabling **Sim-to-Real** transfer and high-throughput **PPO training** with experimentally identified dynamics.

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


## Citation

If you find this codebase useful in your research, please cite both the software repository (for the implementation details) and the accompanying paper (for the methodology).

### **1. Cite this Repository (Software)**:
> *Primary reference for the physics engine plugins, aerodynamic force field implementation, and code usage.*

```bibtex
@software{isaaclab_ball_aero,
  author = {Zhang, Renhong},
  title = {Isaac Lab Ball Aerodynamics},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/renhong-zhang/isaaclab-ball-aerodynamics}}
}

```

### **2. Cite the Paper (Methodology)**:


```bibtex
@misc{hu2025versatilehumanoidtabletennis,
      title={Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation}, 
      author={Muqun Hu and Wenxi Chen and Wenjing Li and Falak Mandali and Zijian He and Renhong Zhang and Praveen Krisna and Katherine Christian and Leo Benaharon and Dizhi Ma and Karthik Ramani and Yan Gu},
      year={2025},
      eprint={2509.21690},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.21690}, 
}

```
