import os
import queue
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.parkour_agent import (
    ParkourAgent,
    ParkourStandAgent,
)
from instinct_onboard.joystick import UnitreeJoyStick
from instinct_onboard.ros_nodes.unitree import UnitreeNode
from instinct_onboard.ros_nodes.zmq_camera import ZMQCamera

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500

"""
G1 Parkour Node — OFF-BOARD (laptop) variant of g1_parkour.py.

Identical state machine and controls to g1_parkour.py, but designed to run on
an external computer wired to the G1 (192.168.123.x):

- Depth comes from the Jetson's ZMQ depth broadcast (g1_depth_stream.service,
  raw z16 848x480 @ ~30fps on tcp://192.168.123.164:5555) via ZMQCamera,
  instead of a locally-attached RealSense. Frames are converted to metres and
  resized to 480x270 so the policy input pipeline matches the onboard script.
- /lowstate, /secondary_imu, /wirelesscontroller, /lowcmd flow over DDS
  (CycloneDDS bound to the robot NIC — source setup_env.sh first).

NOTE on depth_fps: the depth history buffer is appended once per 50 Hz main
loop tick (not per camera frame), so depth_fps=50 makes the agent's history
subsampling interval exactly match the simulation's 0.1 s frame spacing.

Joystick: any button to init -> cold start runs -> R1 = stand -> L1 = parkour
-> R1 = back to stand. L2 or R2 = emergency stop (motors off + exit) anytime.

Example:
    python g1_parkour_laptop.py --logdir /path/to/parkour_onboard_preview_stair \
        --standdir /path/to/stand_onboard            # dryrun (default)
    ... --nodryrun                                   # real motor commands
"""


class G1ParkourLaptopNode(ZMQCamera, UnitreeNode):
    def __init__(
        self,
        *args,
        lin_vel_deadband: float = 0.5,
        ang_vel_deadband: float = 0.5,
        cmd_px_range: tuple = (0.5, 0.5),
        cmd_nx_range: tuple = (0.0, 0.0),
        cmd_py_range: tuple = (0.0, 0.0),
        cmd_ny_range: tuple = (0.0, 0.0),
        cmd_pyaw_range: tuple = (0.0, 1.0),
        cmd_nyaw_range: tuple = (0.0, 1.0),
        joystick_topic: str = "/wirelesscontroller",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None

        # Velocity-control parameters — the script writer decides how velocity
        # is computed from joystick input.
        self._lin_vel_deadband = lin_vel_deadband
        self._ang_vel_deadband = ang_vel_deadband
        self._cmd_px_range = cmd_px_range
        self._cmd_nx_range = cmd_nx_range
        self._cmd_py_range = cmd_py_range
        self._cmd_ny_range = cmd_ny_range
        self._cmd_pyaw_range = cmd_pyaw_range
        self._cmd_nyaw_range = cmd_nyaw_range

        # Wire up the wireless controller (joystick).  The default
        # safety-shutdown (turn off motors + SystemExit) is sufficient.
        self._joystick = UnitreeJoyStick(self, joy_stick_topic=joystick_topic)

        # Generic velocity command buffer — populated each main-loop tick
        # from the joystick.  Agents read it via _get_base_velocity_cmd_obs.
        self.base_velocity_cmd = np.zeros(3, dtype=np.float32)

    def register_agent(self, name: str, agent):
        self.available_agents[name] = agent

    def check_buffers_ready(self):
        """Wait for lowstate/IMU, the first wireless-controller message, and
        the first real depth frame from the ZMQ stream before the main loop."""
        if not super().check_buffers_ready():
            return False
        if self._joystick.data.ly is None:
            return False
        if not self.has_received_depth_frame:
            self.refresh_camera_data()
            if not self.has_received_depth_frame:
                self.get_logger().info(
                    "Waiting for the first depth frame from the ZMQ stream "
                    f"({self.zmq_addr}) — is g1_depth_stream.service running?",
                    throttle_duration_sec=5.0,
                )
                return False
        return True

    def start_ros_handlers(self):
        super().start_ros_handlers()
        # build the joint state publisher and base_link tf publisher
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        # start the main loop with 20ms duration
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_timer_counter: int = 0  # counter for the main loop timer to assess the actual frequency
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)

    def main_loop_callback(self):
        main_loop_callback_start_time = time.time()

        # Compute base velocity command from joystick (or any other source the
        # script writer chooses) and write it to the generic ros_node buffer so
        # the agent can read it via _get_base_velocity_cmd_obs.
        self.base_velocity_cmd = self._compute_velocity_from_joystick()

        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        elif self.current_agent_name == "cold_start":
            tjs, status = self.available_agents[self.current_agent_name].step()
            if status != AgentStatus.Working:
                if "stand" in self.available_agents.keys():
                    self.get_logger().info(
                        "ColdStartAgent done, press 'R1' to switch to stand agent.", throttle_duration_sec=10.0
                    )
                else:
                    self.get_logger().info(
                        "ColdStartAgent done, press any direction button to switch to parkour agent.",
                        throttle_duration_sec=10.0,
                    )
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working and (self._joystick.data.R1):
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "stand":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.refresh_camera_data()
            self.send_target_joint_state(tjs)
            if self._joystick.data.L1:
                self.get_logger().info("L1 button pressed, switching to parkour agent.")
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "parkour":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if self._joystick.data.R1:
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

        # count the main loop timer counter and log the actual frequency every 500 counts
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_callback_time_consumptions.put(time.time() - main_loop_callback_start_time)
            self.main_loop_timer_counter += 1
            if self.main_loop_timer_counter % MAIN_LOOP_FREQUENCY_CHECK_INTERVAL == 0:
                time_consumptions = [
                    self.main_loop_callback_time_consumptions.get() for _ in range(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)
                ]
                self.get_logger().info(
                    f"Actual main loop frequency: {(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL / (time.time() - self.main_loop_timer_counter_time)):.2f} Hz. Mean time consumption: {np.mean(time_consumptions):.4f} s."
                )
                self.main_loop_timer_counter = 0
                self.main_loop_timer_counter_time = time.time()

    def _compute_velocity_from_joystick(self) -> np.ndarray:
        """Compute [vx, vy, yaw] from the wireless controller axes.

        Same mapping as g1_parkour.py.
        """
        jy = self._joystick.data
        # left-y for forward/backward
        ly = jy.ly
        if ly > self._lin_vel_deadband:
            vx = (ly - self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vx = vx * (self._cmd_px_range[1] - self._cmd_px_range[0]) + self._cmd_px_range[0]
        elif ly < -self._lin_vel_deadband:
            vx = (ly + self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vx = vx * (self._cmd_nx_range[1] - self._cmd_nx_range[0]) - self._cmd_nx_range[0]
        else:
            vx = 0.0
        # left-x for side moving left/right
        lx = -jy.lx
        if lx > self._lin_vel_deadband:
            vy = (lx - self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vy = vy * (self._cmd_py_range[1] - self._cmd_py_range[0]) + self._cmd_py_range[0]
        elif lx < -self._lin_vel_deadband:
            vy = (lx + self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vy = vy * (self._cmd_ny_range[1] - self._cmd_ny_range[0]) - self._cmd_ny_range[0]
        else:
            vy = 0.0
        # right-x for turning left/right
        rx = -jy.rx
        if rx > self._ang_vel_deadband:
            yaw = (rx - self._ang_vel_deadband) / (1 - self._ang_vel_deadband)
            yaw = yaw * (self._cmd_pyaw_range[1] - self._cmd_pyaw_range[0]) + self._cmd_pyaw_range[0]
        elif rx < -self._ang_vel_deadband:
            yaw = (rx + self._ang_vel_deadband) / (1 - self._ang_vel_deadband)
            yaw = yaw * (self._cmd_nyaw_range[1] - self._cmd_nyaw_range[0]) - self._cmd_nyaw_range[0]
        else:
            yaw = 0.0
        return np.array([vx, vy, yaw], dtype=np.float32)


def main(args):
    rclpy.init()

    node = G1ParkourLaptopNode(
        zmq_addr=args.zmq_addr,
        zmq_frame_shape=(848, 480),  # raw resolution of the Orin's stream_depth.py broadcast
        depth_resolution=(480, 270),  # presented downstream, same as the onboard script
        # Depth history appends happen per 50Hz main-loop tick; 50 here makes the
        # agent's history subsampling interval match the sim's 0.1s spacing.
        depth_fps=50,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        dryrun=not args.nodryrun,
        lin_vel_deadband=args.lin_vel_deadband,
        ang_vel_deadband=args.ang_vel_deadband,
        cmd_px_range=args.lin_vel_range,
        cmd_pyaw_range=args.ang_vel_range,
        cmd_nyaw_range=args.ang_vel_range,
    )

    stand_agent = ParkourStandAgent(
        logdir=args.standdir,
        ros_node=node,
    )
    node.register_agent("stand", stand_agent)

    parkour_agent = ParkourAgent(
        logdir=args.logdir,
        ros_node=node,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
    )
    node.register_agent("parkour", parkour_agent)

    cold_start_agent = ColdStartAgent(
        startup_step_size=args.startup_step_size,
        ros_node=node,
        joint_target_pos=parkour_agent.default_joint_pos,
        action_terms=parkour_agent.action_terms,
        p_gains=parkour_agent.p_gains * args.kpkd_factor,
        d_gains=parkour_agent.d_gains * args.kpkd_factor,
    )
    node.register_agent("cold_start", cold_start_agent)

    if args.depth_vis or args.pointcloud_vis:
        node.publish_auxiliary_static_transforms("camera_depth_link_transform")

    node.start_ros_handlers()
    node.get_logger().info("G1ParkourLaptopNode is ready to run.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Node shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1 Parkour Node (off-board laptop variant)")
    parser.add_argument(
        "--standdir",
        type=str,
        default=os.environ.get("INSTINCT_STANDDIR"),
        help="Directory to load the stand agent from (default: $INSTINCT_STANDDIR)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.environ.get("INSTINCT_LOGDIR"),
        help="Directory to load the parkour agent from (default: $INSTINCT_LOGDIR)",
    )
    parser.add_argument(
        "--zmq_addr",
        type=str,
        default="tcp://192.168.123.164:5555",
        help="ZMQ depth stream address on the robot's Jetson (default: tcp://192.168.123.164:5555)",
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the cold start agent (default: 0.2)",
    )
    parser.add_argument(
        "--kpkd_factor",
        type=float,
        default=2.0,
        help="KPKD factor for the cold start agent (default: 2.0)",
    )
    parser.add_argument(
        "--depth_vis",
        action="store_true",
        default=False,
        help="Visualize the depth image (default: False)",
    )
    parser.add_argument(
        "--pointcloud_vis",
        action="store_true",
        default=False,
        help="Visualize the pointcloud (default: False)",
    )
    parser.add_argument(
        "--lin_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for linear velocity (default: 0.5)",
    )
    parser.add_argument(
        "--lin_vel_range",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        help="Range of linear velocity, only forward (default: 0.5 0.5)",
    )
    parser.add_argument(
        "--ang_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for angular velocity (default: 0.5)",
    )
    parser.add_argument(
        "--ang_vel_range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Range of angular velocity, both turn left and turn right (default: 0.0 1.0)",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()
    if not args.logdir or not args.standdir:
        parser.error(
            "--logdir and --standdir are required (or export INSTINCT_LOGDIR / INSTINCT_STANDDIR, "
            "e.g. in setup_env.sh)."
        )
    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
