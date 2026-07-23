"""Real-time side-by-side visualization of the RAW RealSense depth image and the
processed depth observation that the parkour policy actually sees.

The processing pipeline mirrors ``ParkourAgent.refresh_depth_frame()``
(instinct_onboard/agents/parkour_agent.py) step by step:

    raw (480x270, metres)
      -> cv2.resize to pattern resolution (e.g. 64x36, nearest)
      -> crop by crop_region -> e.g. 18x32
      -> inpaint pixels closer than 0.2 m
      -> blind-spot zeroing (if configured)
      -> Gaussian blur (if configured)
      -> clip to depth_range and normalize to output_range

Pipeline parameters are parsed from the SAME ``params/env.yaml`` of the
checkpoint (pass ``--logdir``), so what you see is what the policy gets.
Without ``--logdir`` it falls back to the values of
``parkour_onboard_preview_stair``.

This tool talks to the camera directly via pyrealsense2 — it does NOT need
ROS or the robot to be running, and it never sends any command.

Two depth sources are supported:

    --source realsense   open a locally-connected RealSense via pyrealsense2
                         (use this when running ON the robot's Jetson)
    --source zmq         subscribe to the raw z16 depth stream broadcast by
                         gx_loco_deploy's stream_depth.py (g1_depth_stream.service
                         on the G1's Orin, tcp://<orin>:5555, 848x480 uint16 mm)
                         — use this when running on an external laptop.

Usage:
    # on the robot (with a display or X forwarding)
    python instinct_onboard/Tools/real_time_img.py --logdir /path/to/parkour_onboard_preview_stair

    # on the laptop, receiving depth from the Orin over the network
    python instinct_onboard/Tools/real_time_img.py --source zmq --logdir ... --stages

    # headless: dump side-by-side PNGs periodically instead of opening a window
    python instinct_onboard/Tools/real_time_img.py --source zmq --logdir ... --headless --save-dir /tmp/depth_vis

Keys (window mode):  q / ESC quit,  s save a snapshot next to --save-dir (default ./depth_vis)
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import yaml


class DepthObsPipeline:
    """Replica of ParkourAgent's depth processing, parameterized from env.yaml."""

    def __init__(self, logdir: str | None):
        if logdir is not None:
            self._parse_from_env_yaml(os.path.join(logdir, "params", "env.yaml"))
        else:
            # defaults matching checkpoints/parkour_onboard_preview_stair
            self.output_resolution = (64, 36)  # (width, height)
            self.depth_range = (0.0, 2.5)
            self.depth_output_range = (0.0, 1.0)
            self.crop_region = (18, 0, 16, 16)  # (top, bottom, left, right)
            self.blind_spot_crop = None
            self.gaussian_kernel_size = (3, 3)
            self.gaussian_sigma = 1
            print("[real_time_img] no --logdir given, using parkour_onboard_preview_stair defaults")

    def _parse_from_env_yaml(self, env_yaml_path: str):
        # Same keys as ParkourAgent._parse_depth_image_config()
        with open(env_yaml_path) as f:
            cfg = yaml.unsafe_load(f)
        camera_cfg = cfg["scene"]["camera"]
        pipeline_cfg = camera_cfg["noise_pipeline"]

        self.output_resolution = (
            camera_cfg["pattern_cfg"]["width"],
            camera_cfg["pattern_cfg"]["height"],
        )
        self.depth_range = tuple(pipeline_cfg["depth_normalization"]["depth_range"])
        if pipeline_cfg["depth_normalization"]["normalize"]:
            self.depth_output_range = tuple(pipeline_cfg["depth_normalization"]["output_range"])
        else:
            self.depth_output_range = self.depth_range
        self.crop_region = tuple(pipeline_cfg["crop_and_resize"]["crop_region"]) if "crop_and_resize" in pipeline_cfg else None
        self.blind_spot_crop = tuple(pipeline_cfg["blind_spot"]["crop_region"]) if "blind_spot" in pipeline_cfg else None
        if "gaussian_blur" in pipeline_cfg:
            k = pipeline_cfg["gaussian_blur"]["kernel_size"]
            self.gaussian_kernel_size = (k, k)
            self.gaussian_sigma = pipeline_cfg["gaussian_blur"]["sigma"]
        else:
            self.gaussian_kernel_size = None
            self.gaussian_sigma = None
        print(f"[real_time_img] pipeline params from {env_yaml_path}")
        print(
            f"  resize -> {self.output_resolution} (w,h), crop {self.crop_region}, "
            f"blind_spot {self.blind_spot_crop}, gaussian {self.gaussian_kernel_size}/{self.gaussian_sigma}, "
            f"range {self.depth_range} -> {self.depth_output_range}"
        )

    def process(self, depth_m: np.ndarray) -> dict[str, np.ndarray]:
        """Run the full pipeline on a metric depth image.

        Returns a dict of every stage so callers can visualize intermediates.
        Mirrors ParkourAgent.refresh_depth_frame() exactly.
        """
        stages: dict[str, np.ndarray] = {}

        img = cv2.resize(depth_m, self.output_resolution, interpolation=cv2.INTER_NEAREST)
        stages["1_resize"] = img.copy()

        if self.crop_region is not None:
            shape = img.shape
            x1, x2, y1, y2 = self.crop_region
            img = img[x1 : shape[0] - x2, y1 : shape[1] - y2]
        stages["2_crop"] = img.copy()

        mask = (img < 0.2).astype(np.uint8)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        stages["3_inpaint"] = img.copy()

        if self.blind_spot_crop is not None:
            shape = img.shape
            x1, x2, y1, y2 = self.blind_spot_crop
            img[:x1, :] = 0
            img[shape[0] - x2 :, :] = 0
            img[:, :y1] = 0
            img[:, shape[1] - y2 :] = 0
        stages["4_blind_spot"] = img.copy()

        if self.gaussian_kernel_size is not None:
            img = cv2.GaussianBlur(img, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma)
        stages["5_blur"] = img.copy()

        filt_m = np.clip(img, self.depth_range[0], self.depth_range[1])
        filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
        output = filt_norm * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]
        stages["6_normalized"] = output
        return stages


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

def _to_color(img_m: np.ndarray, vmin: float, vmax: float, upscale_to_h: int, label: str) -> np.ndarray:
    """Colormap a single-channel image and upscale (nearest) to a common height."""
    norm = np.clip((img_m - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    img_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
    h, w = color.shape[:2]
    scale = upscale_to_h / h
    color = cv2.resize(color, (int(round(w * scale)), upscale_to_h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(color, f"{label} {img_m.shape[1]}x{img_m.shape[0]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return color


def compose_frame(raw_m: np.ndarray, stages: dict[str, np.ndarray], pipeline: DepthObsPipeline, show_stages: bool, panel_h: int = 360) -> np.ndarray:
    vmin, vmax = pipeline.depth_range
    panels = [_to_color(raw_m, vmin, vmax, panel_h, "raw(m)")]
    if show_stages:
        for name in ("1_resize", "2_crop", "3_inpaint", "4_blind_spot", "5_blur"):
            panels.append(_to_color(stages[name], vmin, vmax, panel_h, name))
    out_lo, out_hi = pipeline.depth_output_range
    panels.append(_to_color(stages["6_normalized"], out_lo, out_hi, panel_h, "final obs"))
    sep = np.full((panel_h, 4, 3), 60, dtype=np.uint8)
    row: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            row.append(sep)
        row.append(p)
    return np.concatenate(row, axis=1)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=str, default=os.environ.get("INSTINCT_LOGDIR"), help="checkpoint dir containing params/env.yaml (pipeline params); defaults to $INSTINCT_LOGDIR if set")
    parser.add_argument("--source", choices=["realsense", "zmq"], default="realsense", help="depth source: local RealSense (on the robot) or the Orin's ZMQ stream (on a laptop)")
    parser.add_argument("--depth_resolution", type=int, nargs=2, default=[480, 270], metavar=("W", "H"), help="RealSense depth stream resolution (default: 480 270, same as g1_parkour.py)")
    parser.add_argument("--fps", type=int, default=60, help="RealSense depth fps (default: 60)")
    parser.add_argument("--camera_serial", type=str, default=None, help="RealSense serial number (default: first device)")
    parser.add_argument("--zmq-addr", type=str, default="tcp://192.168.123.164:5555", help="ZMQ depth stream address (default: G1 Orin)")
    parser.add_argument("--zmq-shape", type=int, nargs=2, default=[848, 480], metavar=("W", "H"), help="ZMQ stream frame shape (default: 848 480, stream_depth.py's setting)")
    parser.add_argument("--zmq-scale", type=float, default=0.001, help="raw uint16 -> metres factor (default: 0.001, D435 depth scale)")
    parser.add_argument("--stages", action="store_true", help="also show every intermediate processing stage")
    parser.add_argument("--headless", action="store_true", help="no window; periodically save PNGs to --save-dir instead")
    parser.add_argument("--save-dir", type=str, default="./depth_vis", help="where snapshots are written (default: ./depth_vis)")
    parser.add_argument("--save-every", type=float, default=1.0, help="headless mode: seconds between saved frames (default: 1.0)")
    args = parser.parse_args()

    pipeline = DepthObsPipeline(args.logdir)

    # -- depth source setup: get_frame() returns a metric float32 image or None --
    if args.source == "realsense":
        import pyrealsense2 as rs  # imported late so --help works without the lib

        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        if args.camera_serial:
            rs_config.enable_device(args.camera_serial)
        rs_config.enable_stream(rs.stream.depth, args.depth_resolution[0], args.depth_resolution[1], rs.format.z16, args.fps)
        profile = rs_pipeline.start(rs_config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        _ = rs_pipeline.wait_for_frames(1000)  # warm up
        print(f"[real_time_img] RealSense started: {args.depth_resolution} @ {args.fps}fps, depth_scale={depth_scale}")

        def get_frame() -> np.ndarray | None:
            frames = rs_pipeline.wait_for_frames(int(2000 / args.fps))
            f = frames.get_depth_frame()
            if f is None:
                return None
            return np.asanyarray(f.get_data(), dtype=np.float32) * depth_scale

        def cleanup():
            rs_pipeline.stop()

    else:  # zmq
        import zmq

        zmq_ctx = zmq.Context()
        sock = zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)  # always deliver only the latest frame
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.RCVTIMEO = 2000
        sock.connect(args.zmq_addr)
        zw, zh = args.zmq_shape
        expected_bytes = zw * zh * 2  # raw z16
        print(f"[real_time_img] subscribing ZMQ depth stream {args.zmq_addr} ({zw}x{zh} z16, x{args.zmq_scale} -> m)")

        def get_frame() -> np.ndarray | None:
            try:
                buf = sock.recv()
            except zmq.Again:
                print("[real_time_img] no ZMQ frame within 2s — is g1_depth_stream.service running?")
                return None
            if len(buf) != expected_bytes:
                print(f"[real_time_img] unexpected frame size {len(buf)} (expected {expected_bytes}), check --zmq-shape")
                return None
            return np.frombuffer(buf, dtype=np.uint16).reshape(zh, zw).astype(np.float32) * args.zmq_scale

        def cleanup():
            sock.close(0)
            zmq_ctx.term()

    os.makedirs(args.save_dir, exist_ok=True)
    win = "depth raw vs processed  (q: quit, s: snapshot)"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_count, t_fps, fps = 0, time.time(), 0.0
    last_save = 0.0
    try:
        while True:
            raw_m = get_frame()
            if raw_m is None:
                continue

            stages = pipeline.process(raw_m)
            canvas = compose_frame(raw_m, stages, pipeline, show_stages=args.stages)

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - t_fps)
                t_fps = now
            cv2.putText(canvas, f"{fps:.1f} fps", (6, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if args.headless:
                if time.time() - last_save >= args.save_every:
                    path = os.path.join(args.save_dir, f"depth_{time.strftime('%H%M%S')}.png")
                    cv2.imwrite(path, canvas)
                    print(f"[real_time_img] saved {path}", end="\r")
                    last_save = time.time()
            else:
                cv2.imshow(win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = os.path.join(args.save_dir, f"snapshot_{time.strftime('%H%M%S')}.png")
                    cv2.imwrite(path, canvas)
                    print(f"[real_time_img] saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\n[real_time_img] stopped.")


if __name__ == "__main__":
    main()
