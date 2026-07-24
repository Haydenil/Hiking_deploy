"""MuJoCo robot impersonator ("bridge") for sim2sim validation of the
instinct_onboard off-board deployment stack.

This process pretends to be a real Unitree G1 so that
``scripts/g1_parkour_laptop.py`` runs COMPLETELY UNMODIFIED against it:

    publishes  /lowstate            (unitree_hg/LowState,  motor q/dq + pelvis IMU)
    publishes  /secondary_imu       (unitree_hg/IMUState,  torso IMU)
    publishes  /wirelesscontroller  (unitree_go/WirelessController, from keyboard)
    subscribes /lowcmd              (unitree_hg/LowCmd) and runs the motor-level
                                    PD:  tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff
    broadcasts depth over ZMQ       (raw z16 mm frames, like the robot Jetson's
                                    stream_depth.py, default tcp://*:5556)

The MuJoCo scene defaults to the gx_loco_deploy stair scene whose 0.30 m tread
/ 0.075 m rise staircase matches what the policy saw in training, and whose
``g1.xml`` mounts a ``depth_camera`` (fovy 58) at the real D435 position with
qpos[7:36] in Unitree low-level motor order — so motor index i maps directly
to qpos[7+i].

SAFETY: run both this bridge and the deploy script with DDS bound to the
loopback interface only (source sim2sim/env_sim.sh) so that `--nodryrun`
commands can physically never reach a real robot.

Keyboard (in the MuJoCo viewer window; letters are avoided because the viewer
binds many of them to visualization toggles):
    Up arrow   : forward ON                 (ly = 1 -> vx = +0.5 m/s)
    Down arrow : STOP (zero all velocities)
    Left/Right : turn left / right (latching)
    7          : R1 button (cold_start -> stand, parkour -> stand)
    6          : L1 button (stand -> parkour)
    E          : L2 button (EMERGENCY STOP test — deploy exits)
    9 / 8      : virtual gantry ON / OFF ("person holding the robot")
    + / -      : teleport one terrain difficulty row up / down (training-
                 terrain scenes with terrain_origins.json; keeps pose, zeroes
                 velocities, re-engages the gantry — press 8 to release)
    * and /    : switch terrain type — next / previous column (wraps around;
                 keypad * required, / works on both keyboards)
    Enter      : any-button pulse (wake up the deploy script's buffer wait)

Usage:
    source sim2sim/env_sim.sh          # terminal 1
    python sim2sim/g1_mujoco_bridge.py
    source sim2sim/env_sim.sh          # terminal 2
    python instinct_onboard/scripts/g1_parkour_laptop.py \
        --zmq_addr tcp://127.0.0.1:5556 --nodryrun
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from std_msgs.msg import Bool
from unitree_go.msg import WirelessController
from unitree_hg.msg import IMUState, LowCmd, LowState

DEFAULT_SCENE = (
    "/home/galbot/Intern/Deploy/gx_loco_deploy-exp-mjlab_with_heightscan_sim2sim/"
    "gear_sonic_depth/data/robot_model/model_data/g1/scene_29dof_stairs.xml"
)
NUM_MOTORS = 29

# Unitree wireless controller bit masks (robot_cfgs.UnitreeWirelessButtons)
BTN_R1 = 0x0001
BTN_L1 = 0x0002
BTN_L2 = 0x0020
BTN_A = 0x0100

# Default standing pose in Unitree motor order (same values the deploy stack
# reads from env.yaml init_state; used only to spawn the sim robot roughly
# upright — cold_start then ramps from wherever the joints are, as on the
# real robot).
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


class G1MujocoBridge(Node):
    def __init__(self, args):
        super().__init__("g1_mujoco_bridge")
        self.args = args

        # ---------------- MuJoCo model ----------------
        if args.scene.endswith(".mjb"):
            self.model = mujoco.MjModel.from_binary_path(args.scene)
        else:
            self.model = mujoco.MjModel.from_xml_path(args.scene)
        # enlarge the offscreen framebuffer for the depth render resolution
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, args.depth_width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, args.depth_height)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = args.timestep

        self.pelvis_bid = self.model.body("pelvis").id
        self.torso_bid = self.model.body("torso_link").id

        # qpos[7+i] / qvel[6+i] <-> Unitree motor i (verified property of g1.xml)
        self.motor_ctrlrange = self.model.actuator_ctrlrange.copy()  # (29, 2)

        # spawn roughly at the default pose, gantry holds the rest
        for jname, q0 in DEFAULT_POSE.items():
            self.data.joint(jname).qpos[0] = q0
        self.data.qpos[0] = args.spawn_x
        self.data.qpos[1] = args.spawn_y
        self.data.qpos[2] = args.spawn_height
        mujoco.mj_forward(self.model, self.data)

        # Until the first /lowcmd arrives, hold all joints at the spawn pose
        # with a weak PD — mimics the real G1's damped "reset" state (L2+A /
        # L2+B) and prevents the limp robot from folding past its joint
        # protection limits while hanging from the gantry.
        self.hold_q = self.data.qpos[7 : 7 + NUM_MOTORS].copy()
        self.hold_kp = 60.0
        self.hold_kd = 2.0
        self.gantry_xy = self.data.qpos[0:2].copy()  # horizontal anchor while gantry is on
        # gantry vertical reference; updated when teleporting between terrain rows
        self.base_height = float(args.spawn_height)

        # terrain difficulty rows (written by gen_training_terrain.py next to the scene)
        self.terrain_origins = None
        self.terrain_level = 0
        self._teleport_target = None  # (row, col) set by +/-, / and * keys; applied in step_sim
        origins_path = os.path.join(os.path.dirname(os.path.abspath(args.scene)), "terrain_origins.json")
        if os.path.exists(origins_path):
            with open(origins_path) as f:
                info = json.load(f)
            self.terrain_origins = np.asarray(info["origins"])  # (rows, cols, 3)
            self.terrain_columns = info.get("columns", [])
            print(f"[bridge] terrain difficulty rows loaded: {self.terrain_origins.shape[0]} levels "
                  f"(keys +/- switch level), columns: {self.terrain_columns}")

        # ---------------- command state ----------------
        self.cmd_lock = threading.Lock()
        self.cmd_q = np.zeros(NUM_MOTORS)
        self.cmd_dq = np.zeros(NUM_MOTORS)
        self.cmd_kp = np.zeros(NUM_MOTORS)
        self.cmd_kd = np.zeros(NUM_MOTORS)
        self.cmd_tau = np.zeros(NUM_MOTORS)
        self.lowcmd_count = 0

        # ---------------- keyboard / joystick state ----------------
        self.joy_ly = 0.0
        self.joy_lx = 0.0
        self.joy_rx = 0.0
        self._turn_until = 0.0  # arrow-key turns are momentary pulses
        self.btn_until: dict[int, float] = {}  # bit -> wallclock expiry
        self._btn_pulse = 1.2  # seconds a key-press keeps the button held
        self.gantry_on = bool(args.gantry)

        # ---------------- ROS pub/sub ----------------
        self.lowstate_pub = self.create_publisher(LowState, "/lowstate", 10)
        self.imu_pub = self.create_publisher(IMUState, "/secondary_imu", 10)
        self.joy_pub = self.create_publisher(WirelessController, "/wirelesscontroller", 10)
        self.lowcmd_sub = self.create_subscription(LowCmd, "/lowcmd", self._lowcmd_cb, 10)
        # test/automation interface: same effect as keys 9/8
        self.gantry_sub = self.create_subscription(Bool, "/sim/gantry", self._gantry_cb, 10)

        # ---------------- depth streaming (ZMQ, like stream_depth.py) ----------------
        self.zmq_ctx = zmq.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.PUB)
        self.zmq_sock.bind(f"tcp://*:{args.zmq_port}")
        self.renderer = mujoco.Renderer(self.model, height=args.depth_height, width=args.depth_width)
        self.renderer.enable_depth_rendering()
        self._depth_period = 1.0 / args.depth_fps
        self._last_depth_t = 0.0

        self._joy_period = 1.0 / 20.0
        self._last_joy_t = 0.0
        self._sim_steps = 0

        self.get_logger().info(
            f"G1 MuJoCo bridge up: scene={args.scene.split('/')[-1]}, "
            f"physics {1/args.timestep:.0f}Hz, depth {args.depth_width}x{args.depth_height}"
            f"@{args.depth_fps}fps on ZMQ :{args.zmq_port}, gantry={'ON' if self.gantry_on else 'OFF'}"
        )

    # ------------------------------------------------------------------
    # /lowcmd -> motor targets
    # ------------------------------------------------------------------
    def _lowcmd_cb(self, msg: LowCmd):
        with self.cmd_lock:
            for i in range(NUM_MOTORS):
                mc = msg.motor_cmd[i]
                self.cmd_q[i] = mc.q
                self.cmd_dq[i] = mc.dq
                self.cmd_kp[i] = mc.kp
                self.cmd_kd[i] = mc.kd
                self.cmd_tau[i] = mc.tau
            self.lowcmd_count += 1

    def _gantry_cb(self, msg: Bool):
        if msg.data and not self.gantry_on:
            self.gantry_xy = self.data.qpos[0:2].copy()
        self.gantry_on = bool(msg.data)
        if not self.gantry_on:
            self.data.xfrc_applied[self.torso_bid][:] = 0.0
            self.data.xfrc_applied[self.pelvis_bid][:] = 0.0
        self.get_logger().info(f"gantry {'ON' if self.gantry_on else 'OFF'} (via /sim/gantry)")

    # ------------------------------------------------------------------
    # physics + publications
    # ------------------------------------------------------------------
    def _current_col(self) -> int:
        return int(np.argmin(np.abs(self.terrain_origins[0, :, 1] - self.data.qpos[1])))

    def _teleport_to_row(self, row: int, col: int | None = None):
        """Move the robot to terrain tile (row, col); col=None keeps the column.

        Keeps the current joint pose, zeroes all velocities, re-engages the
        gantry at the new spot. The deploy stack keeps running seamlessly —
        its observations contain no world position.
        """
        origins = self.terrain_origins
        if col is None:
            col = self._current_col()
        target = origins[row, col]
        self.data.qpos[0] = target[0]
        self.data.qpos[1] = target[1]
        self.data.qpos[2] = target[2] + 0.78
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # upright orientation
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.base_height = float(target[2] + 0.76)
        self.gantry_xy = self.data.qpos[0:2].copy()
        self.gantry_on = True
        self.terrain_level = row
        mujoco.mj_forward(self.model, self.data)
        col_name = self.terrain_columns[col] if col < len(self.terrain_columns) else f"col{col}"
        print(f"[bridge] difficulty level {row}/{origins.shape[0]-1} on '{col_name}', gantry ON "
              f"(press 8 to release)")

    def step_sim(self):
        if self._teleport_target is not None:
            (row, col), self._teleport_target = self._teleport_target, None
            self._teleport_to_row(row, col)
        q = self.data.qpos[7 : 7 + NUM_MOTORS]
        dq = self.data.qvel[6 : 6 + NUM_MOTORS]
        with self.cmd_lock:
            if self.lowcmd_count == 0:
                # damped hold at spawn pose until the deploy stack takes over
                tau = self.hold_kp * (self.hold_q - q) - self.hold_kd * dq
            else:
                tau = self.cmd_kp * (self.cmd_q - q) + self.cmd_kd * (self.cmd_dq - dq) + self.cmd_tau
        tau = np.clip(tau, self.motor_ctrlrange[:, 0], self.motor_ctrlrange[:, 1])
        self.data.ctrl[:] = tau

        if self.gantry_on:
            self._apply_gantry()
        else:
            self.data.xfrc_applied[self.torso_bid][:] = 0.0
            self.data.xfrc_applied[self.pelvis_bid][:] = 0.0

        mujoco.mj_step(self.model, self.data)
        self._sim_steps += 1

        if not rclpy.ok():
            return
        self._publish_lowstate()
        now = time.time()
        if now - self._last_joy_t >= self._joy_period:
            self._publish_joystick(now)
            self._last_joy_t = now
        if now - self._last_depth_t >= self._depth_period:
            self._publish_depth()
            self._last_depth_t = now

    def _apply_gantry(self):
        """'Safety net + loose leash' gantry at the PELVIS (root).

        Dead-zone design: while the robot stands normally it carries its own
        full weight and walks freely — the gantry only intervenes when it
        drops below the catch height (fall) or strays beyond the leash
        radius. This avoids the earlier failure mode where a constant-uplift
        gantry carried ~20% of the body weight, so releasing it (key 8) was
        a sudden load change that toppled the policy.
        Toggle with keys 9 (on) / 8 (off)."""
        z = self.data.qpos[2]
        vz = self.data.qvel[2]
        catch_z = self.base_height - 0.06  # below natural standing height (follows terrain row)
        fz = 2000.0 * max(catch_z - z, 0.0) - (200.0 * vz if z < catch_z else 0.0)
        # Gentle always-on horizontal anchor: releasing it stores no force
        # when the robot stands at the anchor (unlike the vertical uplift,
        # which was the release-cliff problem), and without it the robot
        # slowly drifts on its feet during cold start, so the "Reached"
        # state flickers and single R1 presses are often ignored.
        dxy = self.data.qpos[0:2] - self.gantry_xy
        fx = -120.0 * dxy[0] - 30.0 * self.data.qvel[0]
        fy = -120.0 * dxy[1] - 30.0 * self.data.qvel[1]
        xmat = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        zaxis = xmat[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(zaxis[2], -1.0, 1.0))))
        # upright assist only beyond 12 deg of pelvis tilt ("spotter's hands")
        if tilt > 12.0:
            tilt_torque = 150.0 * np.cross(zaxis, np.array([0.0, 0.0, 1.0]))
            ang_damp = -10.0 * self.data.qvel[3:6]
        else:
            tilt_torque = np.zeros(3)
            ang_damp = np.zeros(3)
        self.data.xfrc_applied[self.pelvis_bid][:3] = [fx, fy, max(fz, 0.0)]
        self.data.xfrc_applied[self.pelvis_bid][3:] = tilt_torque + ang_damp
        # "hand on the torso": mild upright torque only — the waist kp (28.5)
        # is marginally below the torso's gravity gradient, so an unassisted
        # torso sags to a large-angle equilibrium during cold start exactly
        # like a real robot would without a person steadying it.
        torso_xmat = self.data.xmat[self.torso_bid].reshape(3, 3)
        torso_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_bid, torso_vel, 0)
        torso_tilt = 40.0 * np.cross(torso_xmat[:, 2], np.array([0.0, 0.0, 1.0]))
        self.data.xfrc_applied[self.torso_bid][:3] = 0.0
        self.data.xfrc_applied[self.torso_bid][3:] = torso_tilt - 3.0 * torso_vel[:3]

    def _body_local_angvel(self, bid: int) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 1)
        return vel[:3]  # local-frame angular velocity

    def _publish_lowstate(self):
        msg = LowState()
        msg.mode_machine = 5
        msg.tick = self._sim_steps
        q = self.data.qpos[7 : 7 + NUM_MOTORS]
        dq = self.data.qvel[6 : 6 + NUM_MOTORS]
        for i in range(NUM_MOTORS):
            msg.motor_state[i].q = float(q[i])
            msg.motor_state[i].dq = float(dq[i])
        pelvis_quat = self.data.xquat[self.pelvis_bid]  # wxyz
        msg.imu_state.quaternion = [float(x) for x in pelvis_quat]
        msg.imu_state.gyroscope = [float(x) for x in self._body_local_angvel(self.pelvis_bid)]
        self.lowstate_pub.publish(msg)

        imu = IMUState()
        torso_quat = self.data.xquat[self.torso_bid]
        imu.quaternion = [float(x) for x in torso_quat]
        imu.gyroscope = [float(x) for x in self._body_local_angvel(self.torso_bid)]
        self.imu_pub.publish(imu)

    def _publish_joystick(self, now: float):
        if self.joy_rx != 0.0 and now > self._turn_until:
            self.joy_rx = 0.0  # momentary turn pulse expired
            print("[bridge] turn pulse ended, rx=0")
        msg = WirelessController()
        msg.ly = float(self.joy_ly)
        msg.lx = float(self.joy_lx)
        msg.rx = float(self.joy_rx)
        msg.ry = 0.0
        keys = 0
        for bit, until in list(self.btn_until.items()):
            if now < until:
                keys |= bit
            else:
                del self.btn_until[bit]
        msg.keys = keys
        self.joy_pub.publish(msg)

    def _publish_depth(self):
        self.renderer.update_scene(self.data, camera="depth_camera")
        depth_m = self.renderer.render()  # float32 metres
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        self.zmq_sock.send(depth_mm.tobytes())

    # ------------------------------------------------------------------
    # keyboard
    # ------------------------------------------------------------------
    def key_callback(self, keycode: int):
        # NOTE: letter keys are avoided on purpose — the MuJoCo viewer binds
        # many letters to visualization toggles (W=wireframe, R=reflection,
        # S=shadow, ...). Arrow keys and digits 6-9 are free (digit toggles
        # only exist for geom groups 0-5).
        now = time.time()
        c = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if keycode == 265:  # Up arrow
            self.joy_ly = 1.0
            print("[bridge] forward ON  (vx -> +0.5 m/s once in parkour)")
        elif keycode == 264:  # Down arrow
            self.joy_ly = self.joy_lx = self.joy_rx = 0.0
            print("[bridge] STOP (all velocity zeroed)")
        elif keycode == 263:  # Left arrow — 1s turn pulse, then auto-zero
            self.joy_rx = -1.0
            self._turn_until = now + 1.0
            print("[bridge] turn-left pulse (1s)")
        elif keycode == 262:  # Right arrow — 1s turn pulse, then auto-zero
            self.joy_rx = 1.0
            self._turn_until = now + 1.0
            print("[bridge] turn-right pulse (1s)")
        elif c == "7":
            self.btn_until[BTN_R1] = now + self._btn_pulse
            print("[bridge] R1 pressed (7)")
        elif c == "6":
            self.btn_until[BTN_L1] = now + self._btn_pulse
            print("[bridge] L1 pressed (6)")
        elif c == "E":
            self.btn_until[BTN_L2] = now + 0.4
            print("[bridge] L2 (E-STOP) pressed — deploy script will exit")
        elif c == "9":
            self.gantry_on = True
            self.gantry_xy = self.data.qpos[0:2].copy()  # re-anchor at current spot
            print("[bridge] gantry ON")
        elif c == "8":
            self.gantry_on = False
            self.data.xfrc_applied[self.torso_bid][:] = 0.0
            self.data.xfrc_applied[self.pelvis_bid][:] = 0.0
            print("[bridge] gantry OFF — robot on its own")
        elif keycode in (257, 335):  # Enter / keypad Enter
            self.btn_until[BTN_A] = now + 0.4
            print("[bridge] A pressed (wake-up pulse)")
        elif keycode in (61, 334):  # '=' / '+' key or keypad + : harder terrain row
            if self.terrain_origins is None:
                print("[bridge] no terrain_origins.json for this scene — +/- unavailable")
            elif self.terrain_level >= self.terrain_origins.shape[0] - 1:
                print(f"[bridge] already at max difficulty level {self.terrain_level}")
            else:
                self._teleport_target = (self.terrain_level + 1, None)
        elif keycode in (45, 333):  # '-' key or keypad - : easier terrain row
            if self.terrain_origins is None:
                print("[bridge] no terrain_origins.json for this scene — +/- unavailable")
            elif self.terrain_level <= 0:
                print("[bridge] already at min difficulty level 0")
            else:
                self._teleport_target = (self.terrain_level - 1, None)
        elif keycode in (47, 331):  # '/' key or keypad / : previous terrain type (column)
            if self.terrain_origins is None:
                print("[bridge] no terrain_origins.json for this scene — / and * unavailable")
            else:
                ncols = self.terrain_origins.shape[1]
                self._teleport_target = (self.terrain_level, (self._current_col() - 1) % ncols)
        elif keycode == 332:  # keypad * : next terrain type (column)
            if self.terrain_origins is None:
                print("[bridge] no terrain_origins.json for this scene — / and * unavailable")
            else:
                ncols = self.terrain_origins.shape[1]
                self._teleport_target = (self.terrain_level, (self._current_col() + 1) % ncols)

    def close(self):
        self.zmq_sock.close(0)
        self.zmq_ctx.term()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo G1 bridge for instinct_onboard sim2sim")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE, help="MJCF scene (default: gx stair scene, 0.30m tread = training stairs)")
    parser.add_argument("--zmq_port", type=int, default=5556, help="ZMQ depth broadcast port (default 5556; the real robot uses 5555)")
    parser.add_argument("--depth_width", type=int, default=848)
    parser.add_argument("--depth_height", type=int, default=480)
    parser.add_argument("--depth_fps", type=float, default=30.0)
    parser.add_argument("--timestep", type=float, default=0.002, help="physics dt (default 0.002 = 500Hz; lowstate publishes every step)")
    parser.add_argument("--spawn_height", type=float, default=0.76, help="pelvis spawn z; 0.76 puts the feet on the ground at the default pose so the legs bear weight (like the real start procedure)")
    parser.add_argument("--spawn_x", type=float, default=0.0, help="pelvis spawn x (use the value printed by gen_training_terrain.py)")
    parser.add_argument("--spawn_y", type=float, default=0.0, help="pelvis spawn y")
    parser.add_argument("--no-gantry", dest="gantry", action="store_false", default=True, help="start without the virtual gantry")
    parser.add_argument("--headless", action="store_true", help="no viewer window (keyboard unavailable; drive /wirelesscontroller externally)")
    args = parser.parse_args()

    rclpy.init()
    bridge = G1MujocoBridge(args)

    # ROS callbacks on a background thread; physics paced on the main thread.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    dt = args.timestep
    try:
        if args.headless:
            next_t = time.time()
            while rclpy.ok():
                bridge.step_sim()
                next_t += dt
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.time()  # fell behind; don't spiral
        else:
            with mujoco.viewer.launch_passive(bridge.model, bridge.data, key_callback=bridge.key_callback) as viewer:
                print(__doc__.split("Keyboard")[1].split("Usage:")[0])
                next_t = time.time()
                sync_every = max(1, int(1 / 60 / dt))
                while viewer.is_running() and rclpy.ok():
                    bridge.step_sim()
                    if bridge._sim_steps % sync_every == 0:
                        viewer.sync()
                    next_t += dt
                    sleep = next_t - time.time()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        executor.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
