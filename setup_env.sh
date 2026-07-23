# Source this file to set up the laptop deployment environment:
#     source setup_env.sh
# It wires up: ROS2 Humble + unitree msgs + CycloneDDS rmw bound to the
# robot NIC + the project venv + crc_module on PYTHONPATH.

HIKING_WILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$HIKING_WILD_DIR/unitree_ros2/cyclonedds_ws/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Bind DDS to the USB-ethernet adapter that connects to the G1 (192.168.123.x)
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
    <NetworkInterface name="enx6c1ff7cf9f63" priority="default" multicast="default"/>
</Interfaces></General></Domain></CycloneDDS>'

# crc_module.so lives in instinct_onboard/scripts
export PYTHONPATH="$HIKING_WILD_DIR/instinct_onboard/scripts:$PYTHONPATH"

# Default checkpoint paths — entry scripts and real_time_img read these,
# so --logdir/--standdir can be omitted on the command line.
export INSTINCT_LOGDIR="$HIKING_WILD_DIR/hiking-in-the-wild_Data&Model/data&model/checkpoints/parkour_onboard_preview_stair"
export INSTINCT_STANDDIR="$HIKING_WILD_DIR/hiking-in-the-wild_Data&Model/data&model/checkpoints/stand_onboard"

source "$HIKING_WILD_DIR/.venv/bin/activate"

echo "[setup_env] ready: ROS2 humble + unitree msgs + cyclonedds(enx6c1ff7cf9f63) + venv(python $(python --version 2>&1 | cut -d' ' -f2))"
