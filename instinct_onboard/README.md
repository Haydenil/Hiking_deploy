# Instinct Onboard

Onboard / off-board deployment code for Project Instinct. Designed to run exported
ONNX policies (trained with the InstinctLab / Isaac Lab workflow) on real robots.

The key design principle is **config-as-interface**: the robot-side code does not
hard-code any observation or action definition. Instead it parses the same
`params/env.yaml` that was exported together with the policy, and rebuilds the
exact observation pipeline (term order, scales, clips, history stacking) and
action pipeline (per-joint affine scaling, PD gains) that the policy saw in
simulation. Swapping policies is therefore mostly a matter of pointing
`--logdir` at a different checkpoint directory.

***NOTE*** Currently tested on Ubuntu 22.04 + ROS2 Humble with:
- **Onboard mode**: Unitree G1 (29-DoF) Jetson Orin NX, RealSense D435/D435i
- **Off-board mode**: an x86 laptop wired to the G1's 192.168.123.x network

## Deployment Modes

```
Mode A — ONBOARD (scripts/g1_parkour.py)
  Jetson Orin NX runs everything. RealSense is opened locally via USB
  (pyrealsense2, in a camera subprocess with shared memory).

Mode B — OFF-BOARD (scripts/g1_parkour_laptop.py)
  An external computer runs inference:
    depth   <── ZMQ raw-z16 stream broadcast by the robot's Jetson
    state   <── /lowstate, /secondary_imu, /wirelesscontroller over DDS (CycloneDDS)
    command ──> /lowcmd over DDS (robot's motor-level PD executes it)
  Only the camera source differs from Mode A (ZMQCamera vs RealsenseMPCamera);
  the agents, observation/action pipelines and the state machine are identical.
```

Data flow (both modes):

```
env.yaml ──► OnboardAgent: build obs funcs + action terms
/lowstate ─► joint_pos/joint_vel (sim joint order, via joint_map & joint_signs)
IMU ───────► quat / ang_vel / projected_gravity
joystick ──► base_velocity_cmd [vx, vy, yaw]
depth ─────► resize/crop/inpaint/blur/normalize ──► depth encoder ONNX
                     ▼
             actor ONNX ──► raw action (29,)
                     ▼
   JointPositionAction: q_target = raw * scale + default_joint_pos
                     ▼
   TargetJointState {position, velocity, effort, kp, kd}
                     ▼
   RealNode.send_target_joint_state: NaN check → gain clip → torque-limit
   position clipping → sim→real joint remap → LowCmd + CRC → /lowcmd
                     ▼
   motor-level PD (runs inside the robot, ~kHz):
   tau = kp*(q_des − q) + kd*(dq_des − dq) + tau_ff
```

## Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10 (must match the Python your ROS2 distribution binds to)
- A checkpoint directory with the exported layout:
  ```
  <logdir>/
  ├── exported/actor.onnx              # policy MLP
  ├── exported/0-depth_encoder.onnx    # depth encoder (perceptive policies)
  └── params/env.yaml                  # environment config from training
  ```

## Installation — Mode A (Unitree G1 Jetson Orin NX)

- JetPack
    ```bash
    sudo apt-get update
    sudo apt install nvidia-jetpack
    ```

- Install crc module

    Follow the instruction of [crc_module](https://github.com/ZiwenZhuang/g1_crc) and copy the
    product (`crc_module.so`) to where you launch the python script (e.g. `scripts/`).

- Install `unitree_hg` and `unitree_go` message definitions
  (see [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2))

## Installation — Mode B (off-board computer)

All steps below are on the external computer. Nothing needs to be installed on
the robot beyond what it already runs.

1. **Robot-side requirement (usually already present)**: a depth broadcaster on
   the robot's Jetson that publishes raw `z16` (uint16, millimetres) frames over
   a ZMQ PUB socket, e.g. 848x480 @ 30fps on `tcp://0.0.0.0:5555`. A minimal
   example:
   ```python
   import pyrealsense2 as rs, numpy as np, zmq
   sock = zmq.Context().socket(zmq.PUB); sock.bind("tcp://0.0.0.0:5555")
   p, c = rs.pipeline(), rs.config()
   c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
   p.start(c)
   while True:
       f = p.wait_for_frames().get_depth_frame()
       if f: sock.send(np.asanyarray(f.get_data()).tobytes())
   ```

2. **Python virtual environment** — must be the same Python minor version as
   your ROS2 distro (3.10 for Humble) and must see the system ROS packages:
   ```bash
   # with uv (recommended)
   uv venv --python /usr/bin/python3.10 --system-site-packages .venv
   uv pip install -p .venv/bin/python numpy opencv-python pyyaml pyzmq onnxruntime \
       prettytable numpy-quaternion transformations "empy==3.3.2" transforms3d pybind11
   # install this repo (deps installed above; ros2_numpy comes from colcon, not PyPI)
   uv pip install -p .venv/bin/python --no-deps -e ./instinct_onboard
   ```

3. **Unitree messages + CycloneDDS rmw** (colcon workspace; the from-source
   route below needs no sudo):
   ```bash
   git clone https://github.com/unitreerobotics/unitree_ros2
   cd unitree_ros2/cyclonedds_ws/src
   git clone -b humble          https://github.com/ros2/rosidl_dds          # rosidl_generator_dds_idl
   git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds
   git clone -b humble          https://github.com/ros2/rmw_cyclonedds
   git clone -b humble          https://github.com/Box-Robotics/ros2_numpy
   git clone                    https://github.com/DLu/tf_transformations
   cd ..
   colcon build --packages-select cyclonedds                      # ROS NOT sourced
   source /opt/ros/humble/setup.bash
   colcon build --packages-select rosidl_generator_dds_idl        # build first, alone
   source install/setup.bash
   colcon build --packages-select unitree_api unitree_go unitree_hg \
       rmw_cyclonedds_cpp ros2_numpy tf_transformations
   ```

4. **crc_module for x86** — build [g1_crc](https://github.com/ZiwenZhuang/g1_crc)
   against the venv Python and copy `crc_module.so` (+ `.pyi`) into `scripts/`:
   ```bash
   git clone https://github.com/ZiwenZhuang/g1_crc && cd g1_crc
   mkdir build && cd build
   cmake -Dpybind11_DIR=$(../../.venv/bin/python -m pybind11 --cmakedir) ..
   make && cp ../crc_module.so ../crc_module.pyi <repo>/scripts/
   ```

5. **Environment setup script** — create a `setup_env.sh` you `source` in every
   terminal (adjust the workspace path and the NIC name connected to the robot):
   ```bash
   source /opt/ros/humble/setup.bash
   source <path>/unitree_ros2/cyclonedds_ws/install/setup.bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
       <NetworkInterface name="YOUR_NIC_TO_ROBOT" priority="default" multicast="default"/>
   </Interfaces></General></Domain></CycloneDDS>'
   export PYTHONPATH=<repo>/scripts:$PYTHONPATH     # for crc_module.so
   # optional: default checkpoint paths so --logdir/--standdir can be omitted
   export INSTINCT_LOGDIR=<path>/checkpoints/<parkour_ckpt>
   export INSTINCT_STANDDIR=<path>/checkpoints/<stand_ckpt>
   source <path>/.venv/bin/activate
   ```
   Sanity check afterwards: `ros2 topic hz /lowstate` should show ~1000 Hz.

## Installation (Common)

- Make sure mcap storage for ros2 installed (for `scripts/rosbag.sh` recording)
    ```bash
    sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
    ```

- Install onboard python packages with automatic GPU detection (Mode A pip route)
    ```bash
    pip install -e .            # auto-detects GPU → onnxruntime / onnxruntime-gpu
    pip install -e .[all]       # with OpenCV
    pip install -e .[noopencv]  # without OpenCV deps
    ```
    - Use `FORCE_CPU=1` / `FORCE_GPU=1` to override ONNX Runtime detection.
    - Make sure `import cv2` works in your environment.

## Running (G1 parkour example)

Both entry scripts share the same state machine and controls; pick the one for
your mode:

```bash
cd scripts
# Mode A (on the Jetson):
python g1_parkour.py        --logdir /path/to/parkour_ckpt --standdir /path/to/stand_ckpt
# Mode B (on the laptop):
python g1_parkour_laptop.py --logdir /path/to/parkour_ckpt --standdir /path/to/stand_ckpt \
    [--zmq_addr tcp://192.168.123.164:5555]
```

In Mode B, `--logdir` / `--standdir` default to the `$INSTINCT_LOGDIR` /
`$INSTINCT_STANDDIR` environment variables (export them in your
`setup_env.sh`), so the everyday command shortens to:

```bash
python g1_parkour_laptop.py             # dryrun
python g1_parkour_laptop.py --nodryrun  # command the real motors
```
Explicit command-line paths always override the environment variables.

**Dryrun first.** Without `--nodryrun` (the default) the full pipeline runs but
motor commands go to a renamed topic `/lowcmd_dryrun_<rand>` — the robot never
moves. Verify before powering the motors:

```bash
ros2 topic list | grep dryrun            # find the topic name
ros2 topic hz   /lowcmd_dryrun_XXXXX     # expect a stable 50 Hz
ros2 topic echo /lowcmd_dryrun_XXXXX --once   # sane q / kp / kd, no NaN
```

Only then add `--nodryrun` to command the real motors.

**Operation sequence (G1):**

1. Robot lies flat. On the wireless controller: `L2+R2` (stop sport mode),
   then `L2+A`, `L2+B` (reset/damp joints).
2. Launch the script; press **any button** on the controller once so its first
   message arrives (the script blocks until lowstate + IMU + joystick + first
   depth frame are all received).
3. Cold start runs automatically: joints ramp slowly to the default pose —
   hold the robot upright by hand.
4. **R1** → stand policy (let go gradually). **L1** → parkour policy.
   **R1** again → back to stand.
5. Velocity control in parkour: left stick forward = forward velocity,
   right stick = yaw. Ranges/deadbands via `--lin_vel_range` etc.
6. **Emergency stop: L2 or R2 at any moment** — all motors are disabled
   immediately (the robot goes limp — have a person or gantry ready) and the
   process exits.

## Tools

- `instinct_onboard/Tools/real_time_img.py` — side-by-side visualization of the
  RAW depth image and the exact processed observation the policy receives
  (resize → crop → inpaint → blind-spot → blur → normalize, parameters parsed
  from the checkpoint's `env.yaml`). Two sources:
  ```bash
  real_time_img --logdir <ckpt> --stages                 # local RealSense (on the robot)
  real_time_img --source zmq --logdir <ckpt> --stages    # ZMQ stream (on the laptop)
  # --headless --save-dir DIR to dump PNGs instead of opening a window
  ```
  (installed as a console script by `pip install -e .`; `--logdir` also
  defaults to `$INSTINCT_LOGDIR`, so with the env var set this shortens to
  `real_time_img --source zmq --stages`)

## Code Structure Introduction

### ROS nodes (`instinct_onboard/ros_nodes/`)

- `base.py` — `RealNode`: hardware-agnostic command interface. Owns the safety
  chain: NaN rejection, kp/kd clipping, torque-limit position clipping (the PD
  runs in the motors, so the *target position* is clipped such that the implied
  torque stays within limits), and `last_sent_target_joint_state` caching for
  the `last_action` observation.
- `unitree.py` — `UnitreeNode`: `/lowstate` & IMU subscription, sim↔real joint
  reordering (`joint_map`) and sign flips (`joint_signs`), `LowCmd` publishing
  with CRC, out-of-range joint protection (motors off + exit).
- `camera_base.py` — camera mixin abstraction; `CameraProcessSpawner` runs any
  camera in a subprocess with a shared-memory frame exchange.
- `realsense.py` — local RealSense implementations (single- and multi-process).
- `zmq_camera.py` — `ZMQCamera`: consumes a remote raw-z16 ZMQ depth stream
  (CONFLATE, non-blocking) and converts to metric float frames; drop-in
  replacement for `RealsenseMPCamera` in off-board mode.
- To avoid diamond inheritance, each function-specific ROS node is a dedicated
  Mixin class. Inherit everything you need — plus the state-machine logic — in
  your main-entry script (in `scripts/`).

### Agents (`instinct_onboard/agents/`)

- `base.py` — `OnboardAgent`: parses `env.yaml` into observation functions
  (name → `_get_<func>_obs` reflection, clip/scale/history handling) and action
  terms; `ColdStartAgent` ramps joints to a target pose reusing the main
  agent's action-term layout so `last_action` stays consistent across agent
  switches.
- `action_term.py` — Isaac Lab-style action terms (`JointPositionAction` etc.):
  `target = raw * scale + offset`, plus packing/inversion between raw policy
  vectors and full-size `TargetJointState`s.
- `parkour_agent.py` — perceptive parkour agent: proprio + depth-encoder ONNX
  inference, and the depth image pipeline replicating the training-time camera
  noise pipeline.
- Do NOT scale the action of the network output inside an agent. The action
  scaling happens in the action terms; safety clipping happens in the ros node.

### Scripts (`scripts/`)

- Standalone entry points only — never import them from other modules.
- `g1_parkour.py` (onboard) / `g1_parkour_laptop.py` (off-board): identical
  state machine `cold_start → stand ⇄ parkour`; the only difference is the
  camera mixin and its parameters.
- `rosbag.sh` — record all relevant topics (mcap) for offline analysis.

## Safety Notes

- Dryrun is the default; passing `--nodryrun` is an explicit, deliberate act.
- `L2`/`R2` on the wireless controller are the emergency stop in every entry
  script; a joint position outside its protected range also disables all
  motors and kills the process automatically.
- The e-stop drops all PD gains to zero: the robot collapses. Always have a
  person or a gantry ready to catch it.
- Wearing shoes on the robot (rubber feet covers) is strongly recommended for
  friction and stability.
