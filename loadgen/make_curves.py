"""Load-profile curve generator for autoscaler experiments (Implementation Guide, Section 9.3).

Generates CSV curves (t_seconds, rps) with a seed for the four load profiles:
steady, diurnal, matchday, flashcrowd. Curves are committed with their hash so
runs are reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys

OUT_DIR = os.path.join(os.path.dirname(__file__), "curves")

PROFILES = ["steady", "diurnal", "matchday", "flashcrowd"]

# Duration in seconds for each curve (30 minutes by default).
DURATION_S = 1800
STEP_S = 10

# Matchday profile thresholds (minutes).
MATCHDAY_RAMP_END_MIN = 8
MATCHDAY_SPIKE_END_MIN = 14
MATCHDAY_SPIKE_WIDTH_PCT = 45
MATCHDAY_DIP_START_MIN = 17
MATCHDAY_PEAK_END_MIN = 26

# Flashcrowd step window (seconds).
FLASHCROWD_STEP_START_S = 600
FLASHCROWD_STEP_END_S = 900


def _noise(rng: random.Random, base: float, pct: float = 0.05) -> float:
    return base * (1.0 + rng.uniform(-pct, pct))


def steady_curve(rng: random.Random, duration_s: int = DURATION_S) -> list[dict[str, float]]:
    """150 rps with small noise."""
    return [{"t": t, "rps": _noise(rng, 150.0)} for t in range(0, duration_s, STEP_S)]


def diurnal_curve(rng: random.Random, duration_s: int = DURATION_S) -> list[dict[str, float]]:
    """Sinusoid between 60 and 300 over a compressed 15-minute day."""
    day_s = 900.0  # compressed day
    return [
        {
            "t": t,
            "rps": _noise(rng, 180.0 + 120.0 * math.sin(2 * math.pi * t / day_s)),
        }
        for t in range(0, duration_s, STEP_S)
    ]


def matchday_curve(rng: random.Random, duration_s: int = DURATION_S) -> list[dict[str, float]]:
    """Cricket matchday profile: ramp, step, spikes, dip, peak, drop."""
    points: list[dict[str, float]] = []
    for t in range(0, duration_s, STEP_S):
        minute = t / 60.0
        if minute <= MATCHDAY_RAMP_END_MIN:
            base = 60.0 + (200.0 - 60.0) * (minute / MATCHDAY_RAMP_END_MIN)
        elif minute <= MATCHDAY_SPIKE_END_MIN:
            base = 260.0
            # Random spikes 1.6x-2.2x lasting 20-60s, about one every 2 minutes.
            spike_phase = (minute * 10) % 120
            if spike_phase < MATCHDAY_SPIKE_WIDTH_PCT:
                base *= rng.uniform(1.6, 2.2)
        elif minute <= MATCHDAY_DIP_START_MIN:
            base = 80.0  # innings break dip
        elif minute <= MATCHDAY_PEAK_END_MIN:
            base = 80.0 + (380.0 - 80.0) * ((minute - MATCHDAY_DIP_START_MIN) / 9.0)
        else:
            base = 40.0
        points.append({"t": t, "rps": _noise(rng, base)})
    return points


def flashcrowd_curve(rng: random.Random, duration_s: int = DURATION_S) -> list[dict[str, float]]:
    """120, step to 360 at minute 10, held 5 minutes, back to 120."""
    return [
        {
            "t": t,
            "rps": _noise(
                rng, 360.0 if FLASHCROWD_STEP_START_S <= t < FLASHCROWD_STEP_END_S else 120.0
            ),
        }
        for t in range(0, duration_s, STEP_S)
    ]


CURVE_BUILDERS = {
    "steady": steady_curve,
    "diurnal": diurnal_curve,
    "matchday": matchday_curve,
    "flashcrowd": flashcrowd_curve,
}


def generate_curve(name: str, seed: int, duration_s: int = DURATION_S) -> list[dict[str, float]]:
    if name not in CURVE_BUILDERS:
        raise ValueError(f"unknown profile: {name}")
    rng = random.Random(seed)
    return CURVE_BUILDERS[name](rng, duration_s)


def curve_hash(points: list[dict[str, float]]) -> str:
    payload = json.dumps(points, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_curve(name: str, seed: int, out_dir: str = OUT_DIR) -> str:
    points = generate_curve(name, seed)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "rps"])
        writer.writeheader()
        writer.writerows(points)
    h = curve_hash(points)
    hash_path = path + ".sha256"
    with open(hash_path, "w") as f:
        f.write(h)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate autoscaler load curves")
    parser.add_argument("--profile", choices=PROFILES, default="all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--duration-s", type=int, default=DURATION_S)
    args = parser.parse_args(argv)

    profiles = PROFILES if args.profile == "all" else [args.profile]
    for name in profiles:
        path = write_curve(name, args.seed, out_dir=args.out_dir)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
