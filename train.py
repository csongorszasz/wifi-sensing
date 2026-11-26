#!/usr/bin/env python3
"""Train the Wi-Fi activity classifier from one or more recorded sessions.

Evaluation is leave-one-round-out: the model is tested on a whole round it never
saw, which is the honest estimate of how it will do live (random splits of
overlapping windows would leak and look far too good).

Usage:
    .env/bin/python train.py data/session-*.csv
    .env/bin/python train.py data/*.csv --still-speed 0.3    # camera filter threshold
"""
import argparse
import csv
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from wifisense import COLS, FEATURE_NAMES, FS, LONG_S, SHORT_S, window_features

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["empty", "still", "moving"]
HOP = 0.5


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    num = np.array([[float(r[c]) if r[c] != "" else np.nan for c in COLS] for r in rows])
    meta = {
        "round": np.array([int(r["round"]) for r in rows]),
        "label": np.array([r["label"] for r in rows]),
        "active": np.array([int(r["active"]) for r in rows]),
        "cam_present": np.array([float(r["cam_present"]) if r["cam_present"] != "" else np.nan for r in rows]),
        "cam_speed": np.array([float(r["cam_speed"]) if r["cam_speed"] != "" else np.nan for r in rows]),
    }
    return num, meta


def build_windows(paths, still_speed, empty_presence, short_s=SHORT_S, long_s=LONG_S):
    X, y, groups = [], [], []
    dropped = {c: 0 for c in CLASSES}
    kept = {c: 0 for c in CLASSES}
    speed_by_label = {c: [] for c in CLASSES}
    for fi, path in enumerate(paths):
        loaded = load(path)
        if loaded is None:
            continue
        num, meta = loaded
        for rnd in np.unique(meta["round"]):
            for label in CLASSES:
                m = (meta["round"] == rnd) & (meta["label"] == label) & (meta["active"] == 1)
                if m.sum() < FS * (long_s + 2):
                    continue
                arr = num[m]
                cp, cs = meta["cam_present"][m], meta["cam_speed"][m]
                t = arr[:, 0]
                for t_end in np.arange(t[0] + long_s, t[-1] + 1e-9, HOP):
                    lo, hi = np.searchsorted(t, t_end - long_s), np.searchsorted(t, t_end, side="right")
                    win = arr[lo:hi]
                    if len(win) < 0.6 * FS * long_s:
                        continue
                    s_lo = np.searchsorted(t, t_end - short_s)
                    pres, spd = cp[s_lo:hi], cs[s_lo:hi]
                    if not np.all(np.isnan(pres)):
                        seen = spd[pres > 0]
                        if seen.size:
                            speed_by_label[label].append(float(np.median(seen)))
                        if label == "empty" and np.nanmean(pres) > empty_presence:
                            dropped[label] += 1
                            continue
                        if label == "still" and seen.size and np.median(seen) > still_speed:
                            dropped[label] += 1
                            continue
                    X.append(window_features(win, short_s, long_s))
                    y.append(label)
                    groups.append(f"{os.path.basename(path)}#r{rnd}")
                    kept[label] += 1
    return np.array(X), np.array(y), np.array(groups), kept, dropped, speed_by_label


def make_models():
    return {
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=0),
        "random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
    }


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print("      true \\ predicted " + "".join(f"{c:>9}" for c in CLASSES) + "   recall")
    for c, r in zip(CLASSES, cm):
        rec = r[CLASSES.index(c)] / max(r.sum(), 1)
        print(f"      {c:>20} " + "".join(f"{v:>9}" for v in r) + f"   {rec:6.1%}")
    occ_true = np.array([v != "empty" for v in y_true])
    occ_pred = np.array([v != "empty" for v in y_pred])
    mov_true = np.array([v == "moving" for v in y_true])
    mov_pred = np.array([v == "moving" for v in y_pred])
    print(f"      someone present vs empty: {balanced_accuracy_score(occ_true, occ_pred):.1%} balanced accuracy")
    print(f"      moving vs not moving:     {balanced_accuracy_score(mov_true, mov_pred):.1%} balanced accuracy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--still-speed", type=float, default=0.25,
                    help="drop 'still' windows where the camera saw more motion than this")
    ap.add_argument("--empty-presence", type=float, default=0.2,
                    help="drop 'empty' windows where the camera saw a person this fraction of the time")
    ap.add_argument("--short", type=float, default=SHORT_S, help=f"short window seconds (default {SHORT_S:g})")
    ap.add_argument("--long", type=float, default=LONG_S, help=f"long window seconds (default {LONG_S:g})")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "activity.joblib"))
    args = ap.parse_args()

    X, y, groups, kept, dropped, speeds = build_windows(args.csv, args.still_speed, args.empty_presence, args.short, args.long)
    if len(X) == 0:
        sys.exit("No usable windows. Did the recording finish at least one full activity?")
    print(f"Windows ({args.short:g}s/{args.long:g}s, hop {HOP}s): "
          + ", ".join(f"{c} {kept[c]} kept/{dropped[c]} dropped by camera" for c in CLASSES))
    for c in CLASSES:
        if speeds[c]:
            q = np.percentile(speeds[c], [10, 50, 90])
            print(f"  camera motion while '{c}' (person visible): p10 {q[0]:.3f}  median {q[1]:.3f}  p90 {q[2]:.3f}")
    missing = [c for c in CLASSES if kept[c] == 0]
    if missing:
        sys.exit(f"No data for {missing}; record more.")

    ug = np.unique(groups)
    print(f"Rounds: {len(ug)}  ({', '.join(ug)})")
    results = {}
    if len(ug) >= 2:
        print("\nLeave-one-round-out evaluation (chance = 33%):")
        for name, model in make_models().items():
            pred = np.empty_like(y)
            for g in ug:
                te = groups == g
                if len(np.unique(y[~te])) < len(CLASSES):
                    continue
                model.fit(X[~te], y[~te])
                pred[te] = model.predict(X[te])
            ok = pred != ""
            score = balanced_accuracy_score(y[ok], pred[ok])
            results[name] = score
            print(f"\n  {name}: {score:.1%} balanced accuracy")
            print_confusion(y[ok], pred[ok])
        best = max(results, key=results.get)
    else:
        print("\nOnly one round recorded: no honest held-out estimate possible. Record >= 2 rounds.")
        best = "gradient boosting"

    model = make_models()[best]
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        top = np.argsort(model.feature_importances_)[::-1][:8]
        print("\nMost useful signals: " + ", ".join(FEATURE_NAMES[i] for i in top))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "classes": list(model.classes_),
                 "kind": best, "cv_balanced_accuracy": results.get(best),
                 "short_s": args.short, "long_s": args.long,
                 "trained_on": [os.path.basename(p) for p in args.csv],
                 "trained_at": time.strftime("%Y-%m-%d %H:%M")}, args.out)
    print(f"\nSaved {best} model to {args.out}"
          + (f" (held-out balanced accuracy {results[best]:.1%})" if best in results else ""))
    print("Next: .env/bin/python live.py")


if __name__ == "__main__":
    main()
