#!/usr/bin/env python3
"""Wi-Fi motion sensor using per-antenna RSSI between this laptop and the router.

People moving through the space between (and around) the laptop and the router
disturb the radio multipath, which shows up as fluctuations in received signal
strength. This script:

  1. keeps a steady stream of tiny pings to the router so the signal reading
     is refreshed many times per second,
  2. samples the per-antenna signal from `iw dev <iface> station dump`,
  3. calibrates a baseline while the room is still,
  4. flags MOTION when short-term signal variance rises well above baseline.

No root, no extra packages. Usage:
    python3 wifi_motion.py                 # auto-detect interface & router
    python3 wifi_motion.py --log run.csv   # also record raw samples
    python3 wifi_motion.py --help
"""
import argparse
import csv
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque

SIGNAL_RE = re.compile(r"^\s*signal:\s*(-?\d+)\s*(?:\[([^\]]*)\])?", re.M)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def detect_iface():
    m = re.search(r"Interface\s+(\S+)", sh(["iw", "dev"]))
    if not m:
        sys.exit("No wireless interface found (iw dev).")
    return m.group(1)


def detect_gateway(iface):
    m = re.search(r"default via (\S+) dev " + re.escape(iface), sh(["ip", "route"]))
    if not m:
        sys.exit(f"No default route via {iface}; are you connected to Wi-Fi?")
    return m.group(1)


def read_signal(iface):
    """Return list of per-chain dBm values (or [combined] if chains unavailable)."""
    m = SIGNAL_RE.search(sh(["iw", "dev", iface, "station", "dump"]))
    if not m:
        return None
    if m.group(2):
        return [int(x) for x in m.group(2).split(",")]
    return [int(m.group(1))]


def start_pinger(target, interval):
    try:
        return subprocess.Popen(
            ["ping", "-q", "-i", str(interval), target],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def window_score(win):
    """Motion score: mean over antennas of std-dev of signal within the window."""
    if len(win) < 5:
        return 0.0
    chains = list(zip(*win))
    return statistics.fmean(statistics.pstdev(c) for c in chains)


def bar(value, vmax, width):
    n = max(0, min(width, int(round(value / vmax * width)))) if vmax > 0 else 0
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", help="wireless interface (default: auto)")
    ap.add_argument("--target", help="host to ping (default: router)")
    ap.add_argument("--rate", type=float, default=20, help="samples per second (default 20)")
    ap.add_argument("--window", type=float, default=2.0, help="analysis window seconds (default 2)")
    ap.add_argument("--calib", type=float, default=20, help="calibration seconds, keep still (default 20)")
    ap.add_argument("--sensitivity", type=float, default=3.0,
                    help="threshold = baseline + N * baseline spread; lower = more sensitive (default 3)")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to keep MOTION shown after it stops")
    ap.add_argument("--log", help="write raw samples to this CSV file")
    args = ap.parse_args()

    if not shutil.which("iw"):
        sys.exit("`iw` not found; install it (sudo apt install iw).")
    iface = args.iface or detect_iface()
    target = args.target or detect_gateway(iface)
    if read_signal(iface) is None:
        sys.exit(f"{iface} is not associated to an access point.")

    pinger = start_pinger(target, 0.02)
    period = 1.0 / args.rate
    win = deque(maxlen=max(5, int(args.window * args.rate)))
    logf = open(args.log, "w", newline="") if args.log else None
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["t", "chains_dBm", "score", "motion"])

    print(f"Interface {iface}, pinging {target}. Ctrl+C to stop.")
    print(f"Calibrating for {args.calib:.0f}s — please LEAVE THE ROOM or stay completely still...")

    calib_scores = []
    threshold = None
    last_motion = -1e9
    t0 = time.monotonic()
    next_t = t0
    try:
        while True:
            now = time.monotonic()
            sig = read_signal(iface)
            if sig is None:
                print("\nLost association; waiting...", end="", flush=True)
                time.sleep(1)
                continue
            win.append(sig)
            score = window_score(win)
            elapsed = now - t0

            if threshold is None:
                if elapsed > args.window:  # skip the warm-up until the window is full
                    calib_scores.append(score)
                if elapsed >= args.calib and len(calib_scores) >= 10:
                    base = statistics.fmean(calib_scores)
                    spread = statistics.pstdev(calib_scores)
                    # Floor keeps a perfectly flat baseline (1 dB resolution) from
                    # making every tiny wobble count as motion.
                    threshold = max(base + args.sensitivity * spread, base + 0.25, 0.35)
                    print(f"\nBaseline {base:.2f} dB, spread {spread:.2f} → threshold {threshold:.2f} dB\n")
                else:
                    sys.stdout.write(f"\r  calibrating {elapsed:4.1f}s  signal {sig} dBm  score {score:.2f}   ")
                    sys.stdout.flush()
            else:
                if score > threshold:
                    last_motion = now
                motion = now - last_motion < args.hold
                width = max(10, min(40, shutil.get_terminal_size((80, 20)).columns - 60))
                state = "\033[1;31m MOTION \033[0m" if motion else "\033[32m still  \033[0m"
                sig_str = "/".join(f"{s:>3}" for s in sig)
                sys.stdout.write(
                    f"\r{time.strftime('%H:%M:%S')} [{state}] sig {sig_str} dBm  "
                    f"score {score:4.2f} |{bar(score, threshold * 2.5, width)}| thr {threshold:.2f} "
                )
                sys.stdout.flush()
                if logw:
                    logw.writerow([f"{elapsed:.3f}", ";".join(map(str, sig)), f"{score:.3f}", int(motion)])

            next_t += period
            time.sleep(max(0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if pinger:
            pinger.terminate()
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
