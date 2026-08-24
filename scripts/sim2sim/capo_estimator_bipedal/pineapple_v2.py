"""Pineapple V2 two-wheel-legged CAPO estimator wrapper.

Estimator inputs use robot/XML order, independently of policy_index_map::

    L_hip, L_thigh, L_calf, L_wheel,
    R_hip, R_thigh, R_calf, R_wheel
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np

from . import array_utils as au
from .fusion_estimator import ConfigIndex as CI
from .fusion_estimator import FusionEstimatorCore
from .lowlevel_state import MOTOR_NUM, LowlevelState

PINEAPPLE_MOTOR_NUM = MOTOR_NUM


class PineappleV2Estimate:
    """Values returned by :meth:`PineappleV2StateEstimator.update`."""

    __slots__ = (
        "lin_vel_world", "lin_vel_body",
        "lin_vel_world_raw", "lin_vel_body_raw",
        "pos_world", "rpy", "foot_contact", "odom",
    )

    def __init__(self, odom, vel_world_filt, vel_body_filt,
                 vel_world_raw, vel_body_raw):
        self.odom = odom
        self.pos_world = (odom.XPos, odom.YPos, odom.ZPos)
        self.rpy = (odom.RollRad, odom.PitchRad, odom.YawRad)
        self.foot_contact = (odom.FLFootLanded, odom.FRFootLanded)
        self.lin_vel_world = tuple(float(v) for v in vel_world_filt)
        self.lin_vel_body = tuple(float(v) for v in vel_body_filt)
        self.lin_vel_world_raw = tuple(float(v) for v in vel_world_raw)
        self.lin_vel_body_raw = tuple(float(v) for v in vel_body_raw)


class PineappleV2StateEstimator:
    """CAPO preset for Pineapple V2's two wheel-leg contact chains."""

    def __init__(
        self,
        foot_force_threshold: float = -15.0,
        enable_leg_yaw: bool = False,
        enable_slope: bool = False,
        vel_filter_tau: float | Sequence[float] = (0.08, 0.15, 0.08),
        vel_median_window: int = 5,
        vel_scale: Sequence[float] = (1.0, 1.0, 1.0),
        update_rate_hz: float = 200.0,
    ) -> None:
        if update_rate_hz <= 0.0:
            raise ValueError("update_rate_hz must be positive")

        self.update_rate_hz = float(update_rate_hz)
        self.nominal_dt = 1.0 / self.update_rate_hz
        self.core = FusionEstimatorCore(dt=self.nominal_dt)

        status = [0.0] * 100
        status[CI.IndexInOrOut] = 1
        status[CI.IndexStatusOK] = 1
        status[CI.IndexIMUAccEnable] = 1
        status[CI.IndexIMUQuaternionEnable] = 1
        status[CI.IndexIMUGyroEnable] = 1
        status[CI.IndexJointsXYZEnable] = 1
        status[CI.IndexJointsVelocityXYZEnable] = 1
        status[CI.IndexJointsRPYEnable] = int(enable_leg_yaw)
        status[CI.IndexSlopeModeTimeThreshold] = 1.0
        status[CI.IndexSlopeModeAngleThreshold] = math.radians(5.0)
        status[CI.IndexLegFootForceThreshold] = float(foot_force_threshold)
        status[CI.IndexLegMinStairHeight] = 0.08
        status[CI.IndexStairHeightFogotten] = 1200.0
        status[CI.IndexLegOrientationInitialWeight] = 0.001
        status[CI.IndexLegOrientationTimeWeight] = 1000.0
        status[CI.IndexSlopeEstimationEnable] = int(enable_slope)
        self.core.fusion_estimator_status(status)

        self._state = LowlevelState()

        tau = np.atleast_1d(np.asarray(vel_filter_tau, dtype=float))
        if tau.size == 1:
            tau = np.repeat(tau, 3)
        if tau.shape != (3,):
            raise ValueError("vel_filter_tau must be a scalar or length-3 sequence")

        scale = np.asarray(vel_scale, dtype=float)
        if scale.shape != (3,):
            raise ValueError("vel_scale must be a length-3 sequence")

        self.vel_filter_tau = tau
        self.vel_scale = scale
        self.vel_median_window = max(1, int(vel_median_window))
        self._vel_buffer: deque[np.ndarray] = deque(maxlen=self.vel_median_window)
        self._vel_ema: np.ndarray | None = None
        self._last_timestamp_ms: int | None = None

    @staticmethod
    def _vector(name: str, value: Sequence[float], size: int) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} contains a non-finite value")
        return result

    def reset_position(self) -> None:
        status = [0.0] * 100
        status[CI.IndexInOrOut] = 3
        status[CI.IndexStatusOK] = 1
        self.core.fusion_estimator_status(status)

    def reset_filter(self) -> None:
        self._vel_buffer.clear()
        self._vel_ema = None
        self._last_timestamp_ms = None

    def _filter_velocity(self, velocity, timestamp_ms: int) -> np.ndarray:
        scaled = np.asarray(velocity, dtype=float) * self.vel_scale
        self._vel_buffer.append(scaled)
        median = np.median(np.asarray(self._vel_buffer), axis=0)

        if self._last_timestamp_ms is None:
            dt = self.nominal_dt
        else:
            dt = (timestamp_ms - self._last_timestamp_ms) / 1000.0
            if not 0.0 < dt <= 0.1:
                dt = self.nominal_dt
        self._last_timestamp_ms = timestamp_ms

        alpha = np.where(
            self.vel_filter_tau > 0.0,
            dt / (self.vel_filter_tau + dt),
            1.0,
        )
        if self._vel_ema is None:
            self._vel_ema = median.copy()
        else:
            self._vel_ema = alpha * median + (1.0 - alpha) * self._vel_ema
        return self._vel_ema.copy()

    def update(
        self,
        quat: Sequence[float],
        gyro: Sequence[float],
        accel: Sequence[float],
        q8: Sequence[float],
        dq8: Sequence[float],
        tau8: Sequence[float],
        timestamp_ms: int,
    ) -> PineappleV2Estimate:
        """Run one update using eight robot/XML-order motor samples."""
        quat = self._vector("quat", quat, 4)
        gyro = self._vector("gyro", gyro, 3)
        accel = self._vector("accel", accel, 3)
        q8 = self._vector("q8", q8, PINEAPPLE_MOTOR_NUM)
        dq8 = self._vector("dq8", dq8, PINEAPPLE_MOTOR_NUM)
        tau8 = self._vector("tau8", tau8, PINEAPPLE_MOTOR_NUM)
        timestamp_ms = int(timestamp_ms)

        state = self._state
        state.imu.timestamp = timestamp_ms
        state.imu.quaternion = quat.tolist()
        state.imu.gyroscope = gyro.tolist()
        state.imu.accelerometer = accel.tolist()

        for index in range(PINEAPPLE_MOTOR_NUM):
            motor = state.motorState[index]
            motor.q = float(q8[index])
            motor.dq = float(dq8[index])
            motor.tauEst = float(tau8[index])

        odom = self.core.fusion_estimator(state)

        orientation = au.quaternion_normalize(state.imu.quaternion)
        orientation_inv = au.quaternion_conjugate(orientation)
        vel_world_raw = np.array([odom.XVel, odom.YVel, odom.ZVel], dtype=float)
        vel_body_raw = np.asarray(
            au.quaternion_rotate_vector(orientation_inv, vel_world_raw.tolist()),
            dtype=float,
        )
        vel_body_filt = self._filter_velocity(vel_body_raw, timestamp_ms)
        vel_world_filt = np.asarray(
            au.quaternion_rotate_vector(orientation, vel_body_filt.tolist()),
            dtype=float,
        )

        return PineappleV2Estimate(
            odom, vel_world_filt, vel_body_filt, vel_world_raw, vel_body_raw
        )


__all__ = [
    "PineappleV2Estimate",
    "PineappleV2StateEstimator",
]
