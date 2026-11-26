#!/usr/bin/env python3
"""Guided recording session for training the Wi-Fi activity classifier.

The script walks you through rounds of three activities, announcing each one aloud:
  empty   - leave the room (everyone out, close the door if you can)
  still   - be in the room but stay still: sit, lie down, read, scroll your phone
  moving  - walk around the room, move about normally

Each round runs the three in a random order. A short grace period at the start of
each activity (to get into place) is not used for training. The webcam is used
only as a sanity check: it records whether a person is visible and how much they
move, so contradicting moments (a person seen during "empty", big motion during
"still") can be dropped. No images are saved.

Keep the laptop and router exactly where they will be when you use the model.

Usage:
    .env/bin/python collect.py                         # 3 rounds x 3 activities x 2 min (~21 min)
    .env/bin/python collect.py --rounds 4 --phase 150  # more data
    .env/bin/python collect.py --no-camera
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from wifisense import COLS, FS, LinkSampler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUCTIONS = {
    "empty": "Please leave the room now. Come back when you hear the next announcement.",
    "still": "Please stay in the room and keep still. Sitting, lying down, or reading is fine.",
    "moving": "Please walk around the room and move about.",
}


QUIET = False


def say(text):
    print(f"\n\n>>> {text}\n")
    if QUIET:
        return
    sys.stdout.write("\a")
    if shutil.which("spd-say"):
        subprocess.Popen(["spd-say", "-w", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Camera(threading.Thread):
    """Background person detector; exposes latest (present, motion_speed)."""

    def __init__(self, model_path, device=0):
        super().__init__(daemon=True)
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.cv2, self.mp = cv2, mp
        self.lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open webcam")
        self.present, self.speed = 0, 0.0
        self.last_seen = -1e9
        self.running = True

    def run(self):
        prev, prev_t, t0 = None, None, time.monotonic()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                data=self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            res = self.lm.detect_for_video(img, int((now - t0) * 1000))
            if res.pose_landmarks:
                pts = [(p.x, p.y) for p in res.pose_landmarks[0] if (p.visibility or 0) > 0.5]
                self.last_seen = now
                if prev is not None and pts and len(pts) == len(prev):
                    disp = sorted(((a - c) ** 2 + (b - d) ** 2) ** 0.5 for (a, b), (c, d) in zip(pts, prev))
                    inst = disp[len(disp) // 2] / max(now - prev_t, 1e-3)
                    self.speed = 0.7 * self.speed + 0.3 * inst
                prev, prev_t = pts, now
            else:
                prev = None
                self.speed *= 0.7
            # count as present for 1 s after last detection (detector flickers)
            self.present = int(now - self.last_seen < 1.0)

    def stop(self):
        self.running = False
        self.join(timeout=2)
        self.cap.release()
        self.lm.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--phase", type=float, default=120, help="seconds per activity (default 120)")
    ap.add_argument("--grace", type=float, default=15, help="seconds to get into place (default 15)")
    ap.add_argument("--start-delay", type=float, default=10)
    ap.add_argument("--out", help="output CSV (default data/session-<time>.csv)")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--iface")
    ap.add_argument("--quiet", action="store_true", help="no spoken announcements or beeps")
    args = ap.parse_args()
    global QUIET
    QUIET = args.quiet

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out = args.out or os.path.join(HERE, "data", time.strftime("session-%Y%m%d-%H%M%S.csv"))

    cam = None
    if not args.no_camera:
        try:
            cam = Camera(os.path.join(HERE, "models", "pose_landmarker_lite.task"))
            cam.start()
        except Exception as e:  # noqa: BLE001
            print(f"Camera unavailable ({e}); continuing without it.")
            cam = None

    sampler = LinkSampler(args.iface)

    # Build schedule: shuffle each round, never repeat the previous activity back-to-back.
    schedule, last = [], None
    for r in range(args.rounds):
        order = ["empty", "still", "moving"]
        while True:
            random.shuffle(order)
            if order[0] != last:
                break
        schedule += [(r, a) for a in order]
        last = order[-1]

    total = len(schedule) * (args.phase + args.grace) + args.start_delay
    print(f"Recording to {out}")
    print(f"{len(schedule)} activities, about {total / 60:.0f} minutes. Interface {sampler.iface}, "
          f"router {sampler.target}, camera {'on' if cam else 'off'}.")
    print("Tip: between rounds, use different spots for 'still' (desk, sofa, bed) and different "
          "walking paths for 'moving'. Ctrl+C stops early; data recorded so far is kept.")
    print(f"Starting in {args.start_delay:.0f}s...")

    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(COLS + ["round", "label", "active", "phase_t", "cam_present", "cam_speed"])
    period = 1.0 / FS
    n = 0
    try:
        time.sleep(args.start_delay)
        for i, (rnd, label) in enumerate(schedule):
            say(f"Round {rnd + 1}, activity {i % 3 + 1}: {label}. {INSTRUCTIONS[label]}")
            p0 = time.monotonic()
            next_t = p0
            while (pt := time.monotonic() - p0) < args.grace + args.phase:
                row = sampler.read(time.time())
                if row is not None:
                    active = int(pt >= args.grace)
                    cp, cs = (cam.present, round(cam.speed, 4)) if cam else ("", "")
                    w.writerow(row + [rnd, label, active, round(pt, 3), cp, cs])
                    n += 1
                    if n % 10 == 0:
                        status = "get into place" if not active else "recording"
                        camtxt = f" | camera: {'person' if cp else 'nobody'} speed {cs:.2f}" if cam else ""
                        sys.stdout.write(f"\r  [{label:>6}] {status:<14} {pt:5.0f}/{args.grace + args.phase:.0f}s "
                                         f"| signal {row[1]:.0f}/{row[2]:.0f} dBm{camtxt}     ")
                        sys.stdout.flush()
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
            f.flush()
        say("Recording finished. Thank you, you can come back now.")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        f.close()
        sampler.close()
        if cam:
            cam.stop()
    print(f"\nSaved {n} samples to {out}")
    shown = os.path.relpath(out, HERE) if out.startswith(HERE + os.sep) else out
    print(f"Next: .env/bin/python train.py {shown}")


if __name__ == "__main__":
    main()
