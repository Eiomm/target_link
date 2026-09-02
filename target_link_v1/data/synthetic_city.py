"""Synthetic city generator — stand-in for real Didi-style trajectory + traffic data.

The V1 specification (`representationV1.md`) consumes, per sub-link + time window:
  * a fleet of GPS trajectories (already map-matched to links), and
  * the production `mean_speed` traffic state feature.

No such raw dataset exists in this environment (the only available trajectory
corpus is geohash-tokenized, without metric link geometry), so we generate a
city with realistic geometry, driving dynamics and measurement noise.

Design rules (must hold so the V1 ablations stay meaningful):
  * links are polylines with realistic lengths (40–1500 m), so long-link
    segmentation is exercised;
  * per link+window there is a hidden *time-independent spatial modulation* of
    speed (e.g. a persistent bottleneck near the far end of the link) that mean
    speed does NOT reveal but the spatial profile does. This makes the core
    research question genuinely non-trivial in the synthetic world;
  * GPS points are subsampled at 1–5 Hz with gaussian noise, so spatial
    binning/interpolation is genuinely required.

Units: meters, seconds, m/s everywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# link graph
# ---------------------------------------------------------------------------


@dataclass
class Link:
    link_id: int
    # polyline in a local metric frame (x=east, y=north, meters)
    points: np.ndarray  # [M, 2] float32
    base_speed: float  # free-flow-ish speed, m/s
    bottleneck_center: float  # fraction of link length where the hidden
    # spatial modulation is centered; NaN -> no spatial structure
    bottleneck_depth: float  # relative speed drop at the center (0..1)
    bottleneck_width: float  # gaussian width (fraction of length); 0 if none
    length: float  # polyline length, meters (precomputed)
    cum: np.ndarray = field(repr=False, default=None)  # cumulative length per vertex

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])


def _polyline_sample(points: np.ndarray, cum: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Sample a polyline at arc-length positions s (meters). Returns [len(s), 2]."""
    s = np.clip(s, 0.0, cum[-1])
    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(cum) - 2)
    seg_len = np.maximum(cum[idx + 1] - cum[idx], 1e-6)
    w = ((s - cum[idx]) / seg_len)[:, None]
    return points[idx] * (1.0 - w) + points[idx + 1] * w


def sample_polyline_at(link: Link, s: np.ndarray) -> np.ndarray:
    return _polyline_sample(link.points, link.cum, s)


class SyntheticCity:
    """Random but seeded road network + per-(link, window) traffic states."""

    def __init__(self, seed: int = 12345, n_links: int = 220, n_windows: int = 72):
        self.rng = np.random.default_rng(seed)
        self.n_windows = n_windows  # e.g. 72 five-minute windows = 6 hours
        self.window_seconds = 300.0
        self.links: List[Link] = []
        self._build_links(n_links)

    # -- graph ---------------------------------------------------------------
    def _build_links(self, n_links: int) -> None:
        rng = self.rng
        for link_id in range(n_links):
            # link length distribution: many short, few long (realistic tail)
            length = float(np.clip(rng.lognormal(mean=4.6, sigma=0.65), 40.0, 1500.0))
            n_pts = max(2, int(length / rng.uniform(80.0, 180.0))) + 1
            heading = rng.uniform(0.0, 2.0 * math.pi)
            # gentle curves: heading drifts slowly along the link
            drift = np.cumsum(rng.normal(0.0, 0.08, n_pts))
            seg_len = np.full(n_pts - 1, length / (n_pts - 1))
            pts = np.zeros((n_pts, 2), dtype=np.float32)
            for i in range(1, n_pts):
                ang = heading + drift[i - 1]
                pts[i] = pts[i - 1] + seg_len[i - 1] * np.array([math.cos(ang), math.sin(ang)])
            base_speed = float(rng.uniform(7.0, 16.5))  # 25–60 km/h
            has_structure = rng.random() < 0.55  # 55% of links have spatial shape
            bottleneck_center = float(rng.uniform(0.15, 0.85)) if has_structure else float("nan")
            bottleneck_depth = float(rng.uniform(0.35, 0.75)) if has_structure else 0.0
            bottleneck_width = float(np.clip(rng.normal(0.16, 0.04), 0.06, 0.35)) if has_structure else 0.0
            self.links.append(
                Link(
                    link_id=link_id,
                    points=pts,
                    base_speed=base_speed,
                    bottleneck_center=bottleneck_center,
                    bottleneck_depth=bottleneck_depth,
                    bottleneck_width=bottleneck_width,
                    length=0.0,  # filled by __post_init__
                )
            )

    # -- traffic state ---------------------------------------------------------
    def window_level_factor(self, link: Link, window_id: int) -> float:
        """Time-varying congestion level of the whole link in a window.
        This is what mean_speed reflects; it is observable to all models.
        Deterministic function of (link_id, window_id) — no RNG draw — so
        dataset building is order-independent and reproducible."""
        t = window_id / max(self.n_windows - 1, 1)
        rush = 0.55 + 0.30 * math.sin(2.0 * math.pi * t + 0.3)  # congestion wave
        link_phase = math.sin(2.0 * math.pi * (link.link_id * 0.137))
        # hash-based pseudo-noise: stable per (link, window)
        h = ((link.link_id * 2654435761) ^ (window_id * 40503)) % 100003
        noise = (h / 100003.0 - 0.5) * 0.12
        factor = float(np.clip(rush + 0.22 * link_phase + noise, 0.30, 1.15))
        return factor

    def spatial_profile(self, link: Link, s_frac: np.ndarray) -> np.ndarray:
        """Hidden, time-independent spatial modulation of speed along the link.
        Independent of window level — this is the information the fine-grained
        profile can carry beyond mean speed."""
        if np.isnan(link.bottleneck_center):
            return np.ones_like(s_frac)
        gauss = np.exp(-((s_frac - link.bottleneck_center) ** 2) / (2.0 * link.bottleneck_width ** 2))
        return 1.0 - link.bottleneck_depth * gauss

    def point_speed(
        self, link: Link, window_id: int, s_frac: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Ground-truth instantaneous speed at arc-length fractions s_frac."""
        level = self.window_level_factor(link, window_id)
        shape = self.spatial_profile(link, s_frac)
        v = link.base_speed * level * shape
        v = v * (1.0 + rng.normal(0.0, 0.03, size=s_frac.shape))  # driver noise
        return np.clip(v, 0.4, None)

    def generate_trajectory(
        self, link: Link, window_id: int, rng: np.random.Generator, noise_gps_m: float = 6.0
    ) -> Dict[str, Any]:
        """Drive along `link` during `window_id`; return noisy GPS observations.

        Returns dict with:
          obs_t      [P] seconds from window start
          obs_xy     [P, 2] noisy positions (m)
          obs_dist   [P] cumulative *observed* distance (m) — noisy
          true_s     [P] ground-truth arc length on the link (m)
          true_v     [P] ground-truth speed (m/s)
        """
        L = link.length
        t = 0.0
        s = 0.0
        ts: List[float] = []
        ss: List[float] = []
        # integration in small steps; speed depends on position (and implicitly time)
        while s < L and t < self.window_seconds * 2.0:
            v = float(self.point_speed(link, window_id, np.array([s / L]), rng)[0])
            v = max(v, 0.5)
            dt = float(np.clip(rng.uniform(1.0, 5.0), 1.0, 60.0))  # GPS sample interval
            s += v * dt
            t += dt
            ts.append(t)
            ss.append(min(s, L))
        if len(ts) < 2:
            return {}
        ts_arr = np.asarray(ts)
        ss_arr = np.minimum(np.asarray(ss), L)
        true_v = np.gradient(ss_arr, ts_arr)
        true_v = np.clip(true_v, 0.3, None)
        xy = sample_polyline_at(link, ss_arr)
        xy_obs = xy + rng.normal(0.0, noise_gps_m, size=xy.shape)
        dseg = np.linalg.norm(np.diff(xy_obs, axis=0), axis=1)
        obs_dist = np.concatenate([[0.0], np.cumsum(dseg)])
        obs_t = ts_arr - ts_arr[0]
        # trajectories entering late may not finish inside the window; keep only
        # points observed within the window for realism (window = 300 s)
        keep = obs_t <= self.window_seconds
        return {
            "obs_t": obs_t[keep].astype(np.float32),
            "obs_xy": xy_obs[keep].astype(np.float32),
            "obs_dist": obs_dist[keep].astype(np.float32),
            "true_s": ss_arr[keep].astype(np.float32),
            "true_v": true_v[keep].astype(np.float32),
        }


def build_default_city(n_links: int = 220, n_windows: int = 72, seed: int = 12345) -> SyntheticCity:
    return SyntheticCity(seed=seed, n_links=n_links, n_windows=n_windows)
