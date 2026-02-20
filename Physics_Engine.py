"""
Physics engine utilities for pool ball trajectory estimation.

Implements an Extended Kalman Filter (EKF) with state:
  [x, y, vx, vy]
Assumes motion in the x-y plane with constant kinetic friction and no air drag.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(v) + eps)


class EKFConfig:
    def __init__(
        self,
        mu_k: float = 0.05,
        g: float = 9.81,
        q_pos: float = 1e-4,
        q_vel: float = 1e-3,
        r_pos: float = 1e-2,
    ) -> None:
        # kinetic friction coefficient
        self.mu_k = mu_k
        # m/s^2
        self.g = g
        # Process noise for [x, y, vx, vy]
        self.q_pos = q_pos
        self.q_vel = q_vel
        # Measurement noise for [x, y]
        self.r_pos = r_pos


class BallTrajectoryEKF:
    """
    Extended Kalman Filter for 2D ball trajectory with kinetic friction.

    State vector:
        x = [px, py, vx, vy]^T
    Measurement:
        z = [px, py]^T (position only)
    """

    def __init__(self, config: Optional[EKFConfig] = None) -> None:
        self.config = config or EKFConfig()
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 1.0

    def initialize(self, px: float, py: float, vx: float, vy: float, P0: Optional[np.ndarray] = None) -> None:
        self.x = np.array([[px], [py], [vx], [vy]], dtype=float)
        if P0 is None:
            self.P = np.eye(4, dtype=float)
        else:
            self.P = P0.astype(float)

    def _acceleration(self, vx: float, vy: float) -> np.ndarray:
        speed = _safe_norm(np.array([vx, vy]))
        # a = -mu * g * v / |v|
        a = -self.config.mu_k * self.config.g * np.array([vx, vy]) / speed
        # If speed is tiny, clamp acceleration to zero to avoid jitter
        if speed < 1e-4:
            a[:] = 0.0
        return a.reshape((2, 1))

    def _jacobian_F(self, dt: float, vx: float, vy: float) -> np.ndarray:
        """
        Jacobian of the motion model with respect to state.
        For friction acceleration a = -k * v / |v|, k = mu*g
        """
        k = self.config.mu_k * self.config.g
        v = np.array([vx, vy], dtype=float)
        speed = _safe_norm(v)
        if speed < 1e-4:
            # Near zero speed: linearize as constant velocity (no accel)
            F = np.eye(4, dtype=float)
            F[0, 2] = dt
            F[1, 3] = dt
            return F

        vx_, vy_ = v
        s = speed
        # Derivative of v/|v|
        dfdv = np.array(
            [
                [1.0 / s - (vx_ * vx_) / (s ** 3), -(vx_ * vy_) / (s ** 3)],
                [-(vx_ * vy_) / (s ** 3), 1.0 / s - (vy_ * vy_) / (s ** 3)],
            ],
            dtype=float,
        )
        # dv/dt = v + a*dt, a = -k * v/|v|
        # Thus, d(v_next)/d(v) = I - k*dfdv*dt
        F_vv = np.eye(2) - k * dfdv * dt

        # position update: p_next = p + v*dt + 0.5*a*dt^2
        # d(p_next)/d(v) = I*dt + 0.5*da/dv*dt^2, da/dv = -k*dfdv
        F_pv = np.eye(2) * dt - 0.5 * k * dfdv * (dt ** 2)

        F = np.eye(4, dtype=float)
        F[0:2, 2:4] = F_pv
        F[2:4, 2:4] = F_vv
        return F

    def predict(self, dt: float) -> None:
        px, py, vx, vy = self.x.flatten().tolist()
        a = self._acceleration(vx, vy)

        # State prediction
        px_new = px + vx * dt + 0.5 * a[0, 0] * dt * dt
        py_new = py + vy * dt + 0.5 * a[1, 0] * dt * dt
        vx_new = vx + a[0, 0] * dt
        vy_new = vy + a[1, 0] * dt

        self.x = np.array([[px_new], [py_new], [vx_new], [vy_new]], dtype=float)

        # Covariance prediction
        F = self._jacobian_F(dt, vx, vy)
        Q = np.diag(
            [
                self.config.q_pos,
                self.config.q_pos,
                self.config.q_vel,
                self.config.q_vel,
            ]
        )
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        """
        Update with a position measurement z = [px, py].
        """
        z = z.reshape((2, 1)).astype(float)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.eye(2, dtype=float) * self.config.r_pos

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position_velocity(self) -> tuple[float, float, float, float]:
        px, py, vx, vy = self.x.flatten().tolist()
        return px, py, vx, vy

    def step(self, z: Optional[np.ndarray] = None, dt: float = 1.0 / 30.0) -> None:
        """
        Convenience step: predict with dt (default 30 FPS) then update if z provided.
        """
        self.predict(dt)
        if z is not None:
            self.update(z)


class TrackedBall:
    def __init__(
        self,
        track_id: int,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        radius_px: Optional[float] = None,
        age: int = 0,
        missed: int = 0,
    ) -> None:
        self.track_id = track_id
        self.position = position
        self.velocity = velocity
        self.bbox = bbox
        self.radius_px = radius_px
        self.age = age
        self.missed = missed


class MultiBallTrackerConfig:
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        max_missed: int = 8,
        distance_threshold_px: float = 45.0,
        ekf_config: Optional[EKFConfig] = None,
    ) -> None:
        self.dt = dt
        self.max_missed = max_missed
        self.distance_threshold_px = distance_threshold_px
        self.ekf_config = ekf_config or EKFConfig()


class MultiBallTracker:
    """
    Multi-object tracker that assigns stable IDs to balls across frames.

    Uses a per-ball EKF and greedy nearest-neighbor association with distance gating.
    """

    def __init__(self, config: Optional[MultiBallTrackerConfig] = None) -> None:
        self.config = config or MultiBallTrackerConfig()
        self._next_id = 1
        self._tracks: dict[int, BallTrajectoryEKF] = {}
        self._meta: dict[int, TrackedBall] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self._meta.clear()

    @staticmethod
    def _extract_measurements(
        detections: Iterable,
    ) -> Tuple[np.ndarray, list[Optional[Tuple[int, int, int, int]]], list[Optional[float]]]:
        centers = []
        bboxes = []
        radii = []
        for det in detections:
            if hasattr(det, "center"):
                center = det.center
                bbox = getattr(det, "bbox", None)
                radius = getattr(det, "radius_px", None)
            else:
                center = det
                bbox = None
                radius = None
            centers.append(center)
            bboxes.append(bbox)
            radii.append(radius)
        if not centers:
            return np.zeros((0, 2), dtype=float), bboxes, radii
        return np.array(centers, dtype=float), bboxes, radii

    def update(self, detections: Iterable, dt: Optional[float] = None) -> list[TrackedBall]:
        dt = self.config.dt if dt is None else dt
        measurements, bboxes, radii = self._extract_measurements(detections)

        # Predict all tracks
        track_ids = list(self._tracks.keys())
        predictions = []
        for tid in track_ids:
            ekf = self._tracks[tid]
            ekf.predict(dt)
            px, py, vx, vy = ekf.get_position_velocity()
            predictions.append((px, py, vx, vy))

        # Associate detections to tracks (greedy nearest neighbor)
        assigned_tracks = set()
        assigned_dets = set()

        if len(track_ids) > 0 and len(measurements) > 0:
            cost = np.zeros((len(track_ids), len(measurements)), dtype=float)
            for i, (px, py, _, _) in enumerate(predictions):
                dx = measurements[:, 0] - px
                dy = measurements[:, 1] - py
                cost[i, :] = np.sqrt(dx * dx + dy * dy)

            while True:
                min_val = np.min(cost) if cost.size else np.inf
                if min_val > self.config.distance_threshold_px:
                    break
                idx = np.argmin(cost)
                i, j = np.unravel_index(idx, cost.shape)
                if i in assigned_tracks or j in assigned_dets:
                    cost[i, j] = np.inf
                    continue
                assigned_tracks.add(i)
                assigned_dets.add(j)
                cost[i, :] = np.inf
                cost[:, j] = np.inf

                tid = track_ids[i]
                z = measurements[j].reshape((2, 1))
                self._tracks[tid].update(z)

                px, py, vx, vy = self._tracks[tid].get_position_velocity()
                meta = self._meta[tid]
                meta.position = (px, py)
                meta.velocity = (vx, vy)
                meta.bbox = bboxes[j]
                meta.radius_px = radii[j]
                meta.age += 1
                meta.missed = 0

        # Handle unmatched tracks
        for idx, tid in enumerate(track_ids):
            if idx in assigned_tracks:
                continue
            meta = self._meta[tid]
            meta.age += 1
            meta.missed += 1
            px, py, vx, vy = self._tracks[tid].get_position_velocity()
            meta.position = (px, py)
            meta.velocity = (vx, vy)

        # Remove stale tracks
        stale = [tid for tid, meta in self._meta.items() if meta.missed > self.config.max_missed]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._meta.pop(tid, None)

        # Create new tracks for unmatched detections
        for j in range(len(measurements)):
            if j in assigned_dets:
                continue
            px, py = measurements[j]
            ekf = BallTrajectoryEKF(self.config.ekf_config)
            ekf.initialize(px, py, 0.0, 0.0)
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = ekf
            self._meta[tid] = TrackedBall(
                track_id=tid,
                position=(px, py),
                velocity=(0.0, 0.0),
                bbox=bboxes[j],
                radius_px=radii[j],
                age=1,
                missed=0,
            )

        return list(self._meta.values())
