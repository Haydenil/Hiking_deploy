# Environment for sim2sim: same as setup_env.sh but DDS is bound to the
# LOOPBACK interface only, so `--nodryrun` commands can physically never
# reach a real robot. Source this in EVERY terminal used for sim2sim.
#     source sim2sim/env_sim.sh

SIM2SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIM2SIM_DIR/../setup_env.sh"

export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
    <NetworkInterface name="lo" priority="default" multicast="default"/>
</Interfaces><AllowMulticast>true</AllowMulticast></General>
<Discovery><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'

echo "[env_sim] DDS re-bound to LOOPBACK ONLY — safe for --nodryrun against the MuJoCo bridge."
