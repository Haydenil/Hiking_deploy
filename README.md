# Hiking Deploy — "Hiking in the Wild" on Unitree G1

Deployment bundle for running the **Hiking-in-the-Wild** perceptive parkour /
stair-climbing policies on a real Unitree G1 (29-DoF) humanoid — either fully
onboard (Jetson Orin NX) or **off-board from an external laptop** wired to the
robot, with depth streamed over the network and motor commands sent over DDS.

```
depth (RealSense D435i on robot) ──ZMQ──►┐
/lowstate, /secondary_imu, joystick ─DDS─►│  laptop: obs pipeline + ONNX policy
                                          │  (50 Hz)
robot motor-level PD ◄────── /lowcmd ─DDS─┘
```

## Repository Layout

| Path | What it is |
|---|---|
| [`instinct_onboard/`](instinct_onboard/) | The deployment framework (vendored from [project-instinct/instinct_onboard](https://github.com/project-instinct/instinct_onboard), with off-board additions). **Start with its [README](instinct_onboard/README.md)** — full architecture, installation and usage docs live there. |
| [`hiking-in-the-wild_Data&Model/`](hiking-in-the-wild_Data%26Model/) | Deployable checkpoints: **stand** policy and **stair-parkour** policy (`actor.onnx` + `0-depth_encoder.onnx` + the training `env.yaml` each), plus motion reference data. |
| [`setup_env.sh`](setup_env.sh) | One-shot environment setup for the off-board laptop: ROS2 Humble + unitree msgs + CycloneDDS (bound to the robot NIC) + Python venv + default checkpoint paths. Edit the NIC name / paths for your machine. |
| [`sim2sim/`](sim2sim/) | **MuJoCo robot impersonator** for validating the whole deployment stack before touching the real robot: publishes `/lowstate`/IMU, runs the motor PD, streams rendered depth over ZMQ, keyboard acts as the joystick — the deploy script runs unmodified against it (DDS bound to loopback for safety). See [sim2sim/README.md](sim2sim/README.md). |

Not tracked (rebuild locally, see [instinct_onboard/README.md](instinct_onboard/README.md)
"Installation — Mode B"): `.venv/` (Python env), `unitree_ros2/` (colcon
workspace: unitree_hg/go msgs + CycloneDDS + rmw), `g1_crc/` (CRC module
sources; the compiled `crc_module.so` is architecture-specific).

## Additions on top of upstream instinct_onboard

- **`ros_nodes/zmq_camera.py`** — `ZMQCamera`: consumes the robot Jetson's raw
  z16 ZMQ depth broadcast (mm → metres, resolution adaptation), a drop-in
  replacement for the local RealSense camera class.
- **`scripts/g1_parkour_laptop.py`** — off-board entry script: identical state
  machine to the onboard `g1_parkour.py`, camera source swapped to `ZMQCamera`.
- **`Tools/real_time_img.py`** — side-by-side viewer of the RAW depth image vs
  the exact processed observation the policy sees (params parsed from the
  checkpoint's `env.yaml`); supports local RealSense and ZMQ sources.
- Rewritten [instinct_onboard/README.md](instinct_onboard/README.md) covering
  both deployment modes end to end.

## Quick Start (off-board / laptop)

```bash
# one-time: follow "Installation — Mode B" in instinct_onboard/README.md
source setup_env.sh                                  # every new terminal

real_time_img --source zmq --stages                  # sanity-check the depth pipeline
python instinct_onboard/scripts/g1_parkour_laptop.py # DRYRUN: full pipeline, robot never moves
#   verify: ros2 topic hz /lowcmd_dryrun_XXXXX  → stable 50 Hz, sane q/kp/kd
python instinct_onboard/scripts/g1_parkour_laptop.py --nodryrun   # real motors
```

Controls: any button to init → cold-start ramps to the default pose (hold the
robot) → **R1** stand → **L1** parkour (left stick forward, right stick yaw) →
**L2/R2 = emergency stop at any moment** (motors go limp — have a person or
gantry ready).

## Safety

Dryrun is the default everywhere; `--nodryrun` is an explicit act. Read the
Safety Notes in [instinct_onboard/README.md](instinct_onboard/README.md) before
powering the motors.

## Acknowledgements

- Deployment framework: [project-instinct/instinct_onboard](https://github.com/project-instinct/instinct_onboard)
- CRC module: [ZiwenZhuang/g1_crc](https://github.com/ZiwenZhuang/g1_crc)
- Messages / DDS: [unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)
