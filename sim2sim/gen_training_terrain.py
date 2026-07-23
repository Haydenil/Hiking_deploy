"""Generate the OFFICIAL training terrain (InstinctMJ parkour ROUGH_TERRAINS_CFG)
as a MuJoCo scene usable by the sim2sim bridge.

The parkour training terrain is a grid of 8x8 m tiles drawn from 10 sub-terrain
types (perlin rough plane, square gaps, pyramid stairs up/down x normal/high,
sparse/dense boxes, inverted pyramid slope), all overlaid with perlin noise.
This script loads that exact config from the InstinctMJ checkout, runs the
official FiledTerrainGenerator, and exports:

    sim2sim/assets/training_terrain.obj         (merged terrain mesh)
    sim2sim/assets/training_terrain_scene.xml   (terrain + G1, bridge-ready)

plus symlinks into sim2sim/assets/ for the G1 model files (g1.xml resolves its
meshes relative to the scene file's directory).

MUST run with the mjlab virtualenv (needs mjlab + torch + trimesh):

    /home/galbot/mjlab/my_mjlab_project/.venv/bin/python \
        sim2sim/gen_training_terrain.py [--rows 4 --cols 10 --seed 0]

Then view it with the bridge (normal deploy venv):

    source sim2sim/env_sim.sh
    python sim2sim/g1_mujoco_bridge.py --scene sim2sim/assets/training_terrain_scene.xml

Rows follow the training curriculum (difficulty grows with row index); the
robot spawns on the flat platform at the centre of tile (0, 0) — the easiest
row. The PLAY variant of the config is used (obstacle walls disabled).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys

HIKING_WILD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTINCT_MJ_SRC = os.path.join(HIKING_WILD, "InstinctMJ", "src")
GX_G1_DIR = (
    "/home/galbot/Intern/Deploy/gx_loco_deploy-exp-mjlab_with_heightscan_sim2sim/"
    "gear_sonic_depth/data/robot_model/model_data/g1"
)
ASSETS_DIR = os.path.join(HIKING_WILD, "sim2sim", "assets")

SCENE_TEMPLATE = """<mujoco model="g1_29dof official training terrain">
  <include file="debug_axis.xml"/>
  <include file="g1.xml"/>

  <statistic center="0 0 1.0" extent="10.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
  </asset>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>
"""


def load_official_cfg():
    """Load ROUGH_TERRAINS_CFG_PLAY from the InstinctMJ checkout.

    parkour_env_cfg.py is loaded as a standalone module (not through the
    instinct_mj.tasks package) to avoid the tasks registry import chain,
    which requires instinct_rl.
    """
    sys.path.insert(0, INSTINCT_MJ_SRC)
    cfg_path = os.path.join(
        INSTINCT_MJ_SRC, "instinct_mj", "tasks", "parkour", "config", "parkour_env_cfg.py"
    )
    src = open(cfg_path).read()
    # keep only the module up to (and including) the PLAY-cfg block; the rest
    # of the file defines env/mdp classes that need instinct_rl.
    marker = "ROUGH_TERRAINS_CFG_PLAY.num_cols"
    end = src.index("\n", src.index(marker))
    header_end = src.index("ROUGH_TERRAINS_CFG =")
    module_src = src[:header_end] + src[src.index("ROUGH_TERRAINS_CFG =") : end]
    # drop imports that belong to the trimmed-away part of the file
    kept_lines = []
    for line in module_src.splitlines():
        if line.startswith(("from instinct_mj.tasks", "from instinct_mj.rl", "from instinct_mj.envs")):
            continue
        kept_lines.append(line)
    namespace: dict = {"__name__": "parkour_terrain_cfg_extract"}
    exec("\n".join(kept_lines), namespace)
    return namespace["ROUGH_TERRAINS_CFG_PLAY"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=4, help="curriculum rows (difficulty levels), default 4")
    parser.add_argument("--cols", type=int, default=10, help="terrain columns (variety), default 10")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = copy.deepcopy(load_official_cfg())
    cfg.num_rows = args.rows
    cfg.num_cols = args.cols
    cfg.seed = args.seed
    cfg.add_lights = False  # scene template provides its own light

    import mujoco
    import numpy as np

    from instinct_mj.terrains.terrain_generator import FiledTerrainGenerator

    print(f"Generating official training terrain: {cfg.num_rows} rows x "
          f"{len(cfg.sub_terrains)} terrain-type columns (curriculum mode), "
          f"tile {cfg.size[0]}x{cfg.size[1]} m, seed {cfg.seed} ...")
    gen = FiledTerrainGenerator(cfg)

    # Build a spec that already contains the G1 (via a wrapper XML in a dir
    # symlinked to the model assets), then let the official generator add its
    # heightfield tiles to the same spec. Heightfields (not meshes) are what
    # give correct non-convex collisions in MuJoCo.
    scene_dir = os.path.join(ASSETS_DIR, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    for name in ("g1.xml", "debug_axis.xml", "meshes"):
        link = os.path.join(scene_dir, name)
        target = os.path.join(GX_G1_DIR, name)
        if not os.path.exists(target):
            raise FileNotFoundError(target)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    wrapper_path = os.path.join(scene_dir, "g1_wrapper.xml")
    with open(wrapper_path, "w") as f:
        f.write(SCENE_TEMPLATE)

    spec = mujoco.MjSpec.from_file(wrapper_path)
    gen.compile(spec)  # terrain generation happens here (adds hfield tiles)

    origins = np.asarray(gen.terrain_origins)  # (rows, type-cols, 3)
    print(f"terrain grid: {origins.shape[0]} rows x {origins.shape[1]} cols; "
          f"columns are terrain types: {list(cfg.sub_terrains.keys())}")
    spawn = origins[0, 0]  # easiest curriculum row, first terrain type
    print(f"tile(0,0) origin (spawn platform): [{spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.3f}]")

    model = spec.compile()
    mjb_path = os.path.join(ASSETS_DIR, "training_terrain_scene.mjb")
    mujoco.mj_saveModel(model, mjb_path, None)
    print(f"wrote {mjb_path} ({os.path.getsize(mjb_path)/1e6:.1f} MB, "
          f"{model.nhfield} heightfield tiles)")

    spawn_z = spawn[2] + 0.76
    print("\nRun it with:\n  source sim2sim/env_sim.sh\n  python sim2sim/g1_mujoco_bridge.py \\\n"
          f"      --scene {os.path.relpath(mjb_path, HIKING_WILD)} \\\n"
          f"      --spawn_x {spawn[0]:.2f} --spawn_y {spawn[1]:.2f} --spawn_height {spawn_z:.3f}")


if __name__ == "__main__":
    main()
