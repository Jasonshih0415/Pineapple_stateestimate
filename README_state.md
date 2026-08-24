# Pineapple State Estimator

This document describes the state estimator used by Pineapple V2 in MuJoCo sim-to-sim. It primarily covers the newly added CAPO (proprioceptive odometry) implementation, data flow, and usage. The project also retains SIPO, making it easy to compare the two estimation methods using the same trajectory.



## Python API

```python
from capo_estimator_bipedal import PineappleV2StateEstimator

estimator = PineappleV2StateEstimator(
    foot_force_threshold=-15.0,
    enable_leg_yaw=False,
    enable_slope=False,
    vel_filter_tau=(0.08, 0.15, 0.08),
    vel_median_window=5,
    vel_scale=(1.0, 1.0, 1.0),
    update_rate_hz=1.0 / config.simulation_dt,
)

result = estimator.update(
    imu_quat,
    ang_vel_b,
    imu_acc,
    qpos,
    qvel,
    tau_meas,
    timestamp_ms,
)
```

Main outputs:

| Field | Description |
|---|---|
| `result.pos_world` | World position `(x, y, z)` |
| `result.lin_vel_world` | Filtered world-frame velocity |
| `result.lin_vel_body` | Filtered body-frame velocity |
| `result.lin_vel_world_raw` | Raw KF world-frame velocity |
| `result.lin_vel_body_raw` | Raw KF body-frame velocity |
| `result.rpy` | `(roll, pitch, yaw)` |
| `result.foot_contact` | Left and right wheel contact probabilities |
| `result.odom` | Complete odometry state |

## CAPO

Run the following command from the repository root:

```bash
python3 scripts/sim2sim/mujoco_rl_capo.py --seq true --capo true
```

- `--seq true`: Uses the scripted command sequence defined in `pineapple_v2.yaml`.
- `--capo true`: Enables the predefined CAPO configuration and result plotting.
- When `--seq true` is not used, commands are provided through teleoperation.
- Use `--config <yaml-path>` to specify a different sim-to-sim configuration.

The following files are generated after execution:

```text
scripts/sim2sim/logs/capo_results.png
scripts/sim2sim/logs/history_data.png
```

`capo_results.png` includes:

- XY trajectory: CAPO and MuJoCo ground truth
- World-frame `vx`, `vy`
- Body-frame `vx`, `vy`
- Yaw rate
- Base Z height


## SIPO

Run SIPO with:

```bash
python3 scripts/sim2sim/mujoco_rl.py --seq true --sipo true
```

You can also enable both estimators in the CAPO integration script:

```bash
python3 scripts/sim2sim/mujoco_rl_capo.py \
    --seq true \
    --sipo true \
    --capo true
```
