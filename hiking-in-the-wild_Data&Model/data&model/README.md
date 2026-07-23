# Motion Reference and Deployable Checkpoints

## Motion Reference
This section describes the configuration for motion references used in **InstinctLab** during training.

**Configuration Steps:**
1.  **File Placement:**
    Place the `parkour_motion_without_run_retargetted.npz` file and the `parkour_motion_without_run` folder in the same directory.
2.  **Update Config:**
    Open `config/g1/g1_parkour_target_amp_cfg.py`.
3.  **Set Path:**
    Locate the `AmassMotionCfg` class and update the `filtered_motion_selection_filepath` to point to the directory from Step 1.

## Deployable Checkpoints
We provide example deployable checkpoints for the **Stand Policy** and **Stair Walking Policy** (MLP network). All demonstration policies are trained using the InstinctLab framework.

### How to Run
Run the following command to start the deployment script:

```bash
python g1_parkour.py \
    --logdir /path/to/parkour_onboard_preview_stair \
    --standdir /path/to/stand_onboard \
    --nodryrun
```

### Operation Sequence (Start-up Procedure)
Please follow this strict sequence to ensure safety:

1.  **Initial State & Reset:**
    * Let the robot lie flat on the ground.
    * Press **L2 + R2** to stop sport mode.
    * Press **L2 + A**, followed by **L2 + B** to reset the robot's joints.
2.  **Launch:** Start the program using the command above. Press **any button** on the wireless controller to init buffers.
3.  **Stand Up:**
    * Manually hold the robot upright.
    * Press **R1** to engage the standing policy.
4.  **Walk:** After the robot stands stably on its own, press **L1** to switch to the walking policy.
5.  **Control:** Left joystick controls forward linear velocity, right joystick controls angular velocity.

### ⚠️ Safety Notes
**Please prioritize safety when testing on a real robot.**

**Hardware Compatibility Warning:**
This codebase has only been tested on the **Unitree G1 (29-DoF version)**. Individual robots may have different configurations, **especially regarding camera placement**. Please verify your hardware settings carefully before deployment.

* **Emergency Stop:** **L2** and **R2** are mapped to the Emergency Stop.
* **Friction:** We strongly recommend equipping the robot with **shoes** to improve friction and stability.