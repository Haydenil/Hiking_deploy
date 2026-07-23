# sim2sim — validate the deployment stack in MuJoCo before the real robot

`g1_mujoco_bridge.py` is a **robot impersonator**: a MuJoCo simulation that
speaks the exact same interfaces as a real Unitree G1, so that
`scripts/g1_parkour_laptop.py` runs **completely unmodified** against it:

| Real robot | Bridge equivalent |
|---|---|
| `/lowstate` (motors + pelvis IMU, DDS) | published from MuJoCo state at ~500 Hz |
| `/secondary_imu` (torso IMU) | published from the torso body |
| `/wirelesscontroller` (joystick) | published from **keyboard** input |
| motor-level PD in the actuators | `tau = kp(q_des−q) + kd(dq_des−dq) + tau_ff` per step |
| Jetson ZMQ depth broadcast (:5555) | MuJoCo depth render from the `depth_camera` (D435 mount, fovy 58) → raw z16 on **:5556** |
| a person holding the robot | virtual gantry (toggle keys 9/8) |

The default scene is the gx_loco_deploy staircase (**0.30 m tread / 0.075 m
rise — the training stair dimensions**). Point `--scene` at any other MJCF
that includes the same `g1.xml`.

## Safety model

`sim2sim/env_sim.sh` re-binds DDS to the **loopback interface only**, so
running the deploy script with `--nodryrun` can physically never reach a real
robot. Always source it (not `setup_env.sh`) in every sim2sim terminal.

## Run

```bash
# terminal 1 — the simulated robot (viewer window opens)
source sim2sim/env_sim.sh
python sim2sim/g1_mujoco_bridge.py

# terminal 2 — the UNMODIFIED deploy script
source sim2sim/env_sim.sh
python instinct_onboard/scripts/g1_parkour_laptop.py \
    --zmq_addr tcp://127.0.0.1:5556 --nodryrun
```

Then drive it from the **viewer window** keyboard exactly like the real
joystick flow:

Letter keys are mostly avoided — the MuJoCo viewer binds many of them to
visualization toggles (W = wireframe, R = reflection, S = shadow, ...).

| Key | Meaning |
|---|---|
| `Enter` | "any button" pulse (wakes the deploy script's buffer wait) |
| `7` | R1 — cold_start → stand, or parkour → stand |
| `6` | L1 — stand → parkour |
| `↑` | forward ON (vx = +0.5 m/s in parkour) |
| `↓` | STOP — zero all velocity commands |
| `←` / `→` | turn left / right (latching) |
| `E` | L2 — emergency-stop test (deploy exits, robot goes limp) |
| `9` / `8` | virtual gantry on / off ("let go of the robot") |

Suggested sequence: wait for cold start to settle → `7` (stand) → `8`
(release the gantry, see if the stand policy balances) → `6` (parkour) →
`↑` (walk forward towards the stairs).

## Notes / limitations

- The bridge holds all joints at the spawn pose with a weak damped PD until
  the first `/lowcmd` arrives — mimicking the real robot's damped "reset"
  state (`L2+A` / `L2+B`), and required so the limp model doesn't fold past
  the deploy stack's joint-protection limits.
- The gantry anchors the pelvis (vertical spring + horizontal spring +
  gentle upright torque on pelvis and torso). The torso assist exists
  because the waist kp (28.5) is marginally below the torso's gravity
  gradient — without a "steadying hand" the torso sags to a large-angle
  equilibrium during cold start, exactly as a real robot would.
- Depth is rendered noise-free at 848×480@30fps; the deploy-side pipeline
  (mm→m, resize, crop, blur, normalize) is identical to the real path.
- Scene assets live in the gx_loco_deploy checkout (see `DEFAULT_SCENE` in
  the bridge); pass `--scene` to use your own copy.
- Automated end-to-end check (headless): the full
  cold_start → stand → parkour chain has been verified with `--headless`
  plus `ros2 topic pub` button pulses; `/lowcmd` stays at 50 Hz throughout.
