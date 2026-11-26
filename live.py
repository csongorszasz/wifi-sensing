#!/usr/bin/env python3
"""Live Wi-Fi sensing with the trained model (no camera).

Predicts on every sample (20 times a second). Shows two readings:
  MOTION    - is someone moving? Fast (~0.3-0.5 s) and reliable (about 98% in testing).
  IN PATH   - is someone (even still) in the laptop-router path? Less reliable:
              a still person is only detectable when they block the signal path,
              and the empty-room level drifts by a few dB over time.
Plus how long ago motion was last seen.

Usage:
    .env/bin/python live.py
    .env/bin/python live.py --confirm 6 --path-lag 10   # steadier, slower
    .env/bin/python live.py --log live.csv              # also record readings
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import joblib
import numpy as np

from wifisense import FEATURE_NAMES, FS, LinkSampler, window_features

HERE = os.path.dirname(os.path.abspath(__file__))
RED, YEL, GRY, GRN, BOLD, RST = "\033[1;31m", "\033[33m", "\033[90m", "\033[32m", "\033[1m", "\033[0m"


def meter(p, width=10):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def ago(seconds):
    if seconds is None:
        return "never"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "activity.joblib"))
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive predictions needed to switch MOTION (default 2 = 0.1 s); higher = fewer blips")
    ap.add_argument("--path-lag", type=float, default=2.5,
                    help="seconds of averaging for IN PATH (default 2.5); higher = steadier")
    ap.add_argument("--iface")
    ap.add_argument("--log")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("No trained model yet. Run collect.py, then train.py.")
    bundle = joblib.load(args.model)
    if bundle["features"] != FEATURE_NAMES:
        sys.exit("Model was trained with a different feature set; retrain with train.py.")
    model, classes = bundle["model"], list(bundle["classes"])
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # one row at a time: multi-core dispatch is 4x slower
    short_s, long_s = bundle.get("short_s", 3.0), bundle.get("long_s", 10.0)  # older models: 3 s / 10 s
    i_empty, i_moving = classes.index("empty"), classes.index("moving")
    print(f"Model: {bundle['kind']} ({short_s:g}s/{long_s:g}s windows), trained {bundle['trained_at']}. Ctrl+C to stop.")

    sampler = LinkSampler(args.iface)
    period = 1.0 / FS
    alpha = 1.0 - math.exp(-period / max(args.path_lag, 1e-3))
    buf = deque(maxlen=int(FS * (long_s + 2)))
    raw_hist = deque(maxlen=max(1, args.confirm))
    p_path, first = 0.5, True
    moving, in_path, last_motion = False, None, None
    logf = open(args.log, "w", newline="") if args.log else None
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["time", "p_moving", "p_in_path", "motion", "in_path"])

    print(f"Warming up for {long_s:g}s...")
    next_t = time.monotonic()
    try:
        while True:
            row = sampler.read(time.time())
            if row is not None:
                buf.append(row)
            if row is not None and buf[-1][0] - buf[0][0] >= long_s:
                now = time.monotonic()
                p = model.predict_proba(window_features(np.array(buf), short_s, long_s)[None, :])[0]
                p_move = p[i_moving]

                # MOTION switches only when the last N raw predictions all agree
                raw_hist.append(p_move > 0.5)
                if len(raw_hist) == raw_hist.maxlen and all(v != moving for v in raw_hist):
                    moving = not moving

                if first:
                    p_path, first = 1.0 - p[i_empty], False
                p_path += alpha * ((1.0 - p[i_empty]) - p_path)
                if moving:
                    last_motion = now
                    p_path = max(p_path, 0.8)  # someone moving is certainly there
                if p_path > 0.65:
                    in_path = True
                elif p_path < 0.35:
                    in_path = False
                elif in_path is not None and 0.45 < p_path < 0.55:
                    in_path = None

                mot = f"{RED}{BOLD}MOVING   {RST}" if moving else f"{GRN}no motion{RST}"
                pth = {True: f"{YEL}likely  {RST}", False: f"{GRY}unlikely{RST}", None: "unsure  "}[in_path]
                since = None if last_motion is None else now - last_motion
                sys.stdout.write(
                    f"\r{time.strftime('%H:%M:%S')}  MOTION {mot} {meter(p_move)}   "
                    f"IN PATH {pth} {meter(p_path)}   last motion: {'now' if moving else ago(since):<10}"
                )
                sys.stdout.flush()
                if logw:
                    logw.writerow([f"{time.time():.2f}", f"{p_move:.3f}", f"{p_path:.3f}",
                                   int(moving), "" if in_path is None else int(in_path)])
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind: don't try to catch up in a burst
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sampler.close()
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
