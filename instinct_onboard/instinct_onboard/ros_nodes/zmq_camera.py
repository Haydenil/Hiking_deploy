from __future__ import annotations

import numpy as np

from instinct_onboard.ros_nodes.camera_base import CameraBase


class ZMQCamera(CameraBase):
    """Depth camera fed by a remote ZMQ raw-z16 stream instead of local hardware.

    Designed for off-board (laptop) inference against a Unitree G1 whose Jetson
    runs ``gx_loco_deploy/stream_depth.py`` (``g1_depth_stream.service``): that
    service broadcasts raw ``z16`` (uint16, millimetres) depth frames over a
    ZMQ PUB socket. This class subscribes to it and presents the frames through
    the standard :class:`CameraBase` interface, so entry scripts can swap it in
    for :class:`~instinct_onboard.ros_nodes.realsense.RealsenseMPCamera` with no
    other changes::

        class G1ParkourLaptopNode(ZMQCamera, UnitreeNode):
            ...

    Conversion performed per frame (all on the receiving side — the robot's
    stream is consumed as-is):

    1. ``uint16`` millimetres -> ``float32`` metres (``zmq_depth_scale``);
    2. optional nearest-neighbour resize from the stream resolution
       (``zmq_frame_shape``) to ``depth_resolution``, so downstream consumers
       see the same raw resolution the original onboard script used.

    The subscriber socket uses ``CONFLATE`` so only the newest frame is kept;
    :meth:`refresh_camera_data` is non-blocking and simply keeps the previous
    frame when no new one has arrived (the stream runs at ~30 fps while the
    control loop polls at 50 Hz).
    """

    def __init__(
        self,
        *args,
        zmq_addr: str = "tcp://192.168.123.164:5555",
        zmq_frame_shape: tuple[int, int] = (848, 480),  # (width, height) of the raw stream
        zmq_depth_scale: float = 0.001,  # uint16 raw -> metres (D435 depth scale)
        depth_resolution: tuple[int, int] = (480, 270),  # (width, height) presented to consumers
        **kwargs,
    ):
        super().__init__(*args, depth_resolution=depth_resolution, **kwargs)
        self.zmq_addr = zmq_addr
        self.zmq_frame_shape = tuple(zmq_frame_shape)
        self.zmq_depth_scale = float(zmq_depth_scale)
        self._zmq_expected_bytes = self.zmq_frame_shape[0] * self.zmq_frame_shape[1] * 2  # z16
        self._zmq_frame_count = 0

        # Start from a zero frame (like RealsenseMPCamera's zero-initialised
        # shared memory) so get_depth_image() never returns None; entry
        # scripts should still gate startup on has_received_depth_frame.
        self._depth_data = np.zeros(self.depth_resolution[::-1], dtype=np.float32)

        import zmq  # imported here so the module stays importable without pyzmq

        self._zmq = zmq
        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sock.setsockopt(zmq.CONFLATE, 1)  # keep only the newest frame
        self._zmq_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._zmq_sock.connect(self.zmq_addr)

    @property
    def has_received_depth_frame(self) -> bool:
        """True once at least one real frame has arrived from the stream."""
        return self._zmq_frame_count > 0

    def refresh_camera_data(self) -> bool:
        """Non-blocking poll of the ZMQ stream.

        Returns True when a new frame was received and stored into
        ``_depth_data``; False when no new frame is available (the previous
        frame is kept, which is the expected steady-state when the control
        loop runs faster than the stream).
        """
        try:
            buf = self._zmq_sock.recv(self._zmq.NOBLOCK)
        except self._zmq.Again:
            return False

        if len(buf) != self._zmq_expected_bytes:
            self._try_log(
                "warn",
                f"ZMQCamera: unexpected frame size {len(buf)} bytes "
                f"(expected {self._zmq_expected_bytes} for {self.zmq_frame_shape} z16); frame dropped.",
            )
            return False

        w, h = self.zmq_frame_shape
        depth_m = np.frombuffer(buf, dtype=np.uint16).reshape(h, w).astype(np.float32) * self.zmq_depth_scale

        if (w, h) != tuple(self.depth_resolution):
            import cv2

            depth_m = cv2.resize(depth_m, tuple(self.depth_resolution), interpolation=cv2.INTER_NEAREST)

        self._depth_data = depth_m
        self._zmq_frame_count += 1
        if self._zmq_frame_count == 1:
            self._try_log("info", f"ZMQCamera: first depth frame received from {self.zmq_addr}.")
        return True

    def destroy_node(self):
        """Close the ZMQ socket, then chain up."""
        if self._zmq_sock is not None:
            self._zmq_sock.close(0)
            self._zmq_sock = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
        super().destroy_node()
