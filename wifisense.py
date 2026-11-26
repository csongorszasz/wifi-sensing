"""Shared code: Wi-Fi link sampling and feature extraction.

A sample is one row: t, sig_chain0, sig_chain1, tx_rate, rx_rate, tx_packets, tx_retries, rx_packets
"""
import re
import subprocess
import sys

import numpy as np

COLS = ["t", "c0", "c1", "tx_rate", "rx_rate", "tx_pk", "tx_retry", "rx_pk"]
FS = 20.0          # target samples per second
SHORT_S = 1.0      # default short analysis window (seconds); each model stores its own
LONG_S = 3.0       # default long analysis window (seconds)

_SIG = re.compile(r"^\s*signal:\s*(-?\d+)\s*(?:\[([^\]]*)\])?", re.M)
_NUM = {
    "tx_rate": re.compile(r"^\s*tx bitrate:\s*([\d.]+)", re.M),
    "rx_rate": re.compile(r"^\s*rx bitrate:\s*([\d.]+)", re.M),
    "tx_pk": re.compile(r"^\s*tx packets:\s*(\d+)", re.M),
    "tx_retry": re.compile(r"^\s*tx retries:\s*(\d+)", re.M),
    "rx_pk": re.compile(r"^\s*rx packets:\s*(\d+)", re.M),
}


def _sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def detect_iface():
    m = re.search(r"Interface\s+(\S+)", _sh(["iw", "dev"]))
    if not m:
        sys.exit("No wireless interface found (iw dev).")
    return m.group(1)


def detect_gateway(iface):
    m = re.search(r"default via (\S+) dev " + re.escape(iface), _sh(["ip", "route"]))
    if not m:
        sys.exit(f"No default route via {iface}; are you connected to Wi-Fi?")
    return m.group(1)


class LinkSampler:
    """Reads link stats from `iw station dump`; keeps a ping stream running so they stay fresh."""

    def __init__(self, iface=None, target=None, ping_interval=0.02):
        self.iface = iface or detect_iface()
        self.target = target or detect_gateway(self.iface)
        self.pinger = subprocess.Popen(
            ["ping", "-q", "-i", str(ping_interval), self.target],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def read(self, t):
        out = _sh(["iw", "dev", self.iface, "station", "dump"])
        m = _SIG.search(out)
        if not m:
            return None
        chains = [int(x) for x in m.group(2).split(",")] if m.group(2) else [int(m.group(1))]
        if len(chains) == 1:
            chains = chains * 2
        vals = []
        for k in ("tx_rate", "rx_rate", "tx_pk", "tx_retry", "rx_pk"):
            mm = _NUM[k].search(out)
            vals.append(float(mm.group(1)) if mm else np.nan)
        return [t, float(chains[0]), float(chains[1])] + vals

    def close(self):
        self.pinger.terminate()


def _band_powers(x, fs):
    x = x - x.mean()
    if len(x) < 8 or not np.any(x):
        return [0.0, 0.0, 0.0]
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 / len(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    bands = [(0.3, 1.0), (1.0, 3.0), (3.0, fs / 2 + 1e-9)]
    return [float(np.log1p(p[(f >= lo) & (f < hi)].sum())) for lo, hi in bands]


FEATURE_NAMES = (
    [f"c{c}_{n}" for c in (0, 1) for n in
     ("mean", "std", "ptp", "absdiff", "change", "b_slow", "b_mid", "b_fast",
      "L_std", "L_ptp", "L_change", "step")]
    + ["xdiff_std", "xcorr", "tx_rate_mean", "tx_rate_std", "rx_rate_std", "retry_rate", "rx_pps"]
)


def window_features(arr, short_s=SHORT_S, long_s=LONG_S):
    """Features for the window ending at the last row of `arr` (rows cover >= long_s seconds)."""
    t = arr[:, 0]
    long_ = arr[t >= t[-1] - long_s]
    short = arr[t >= t[-1] - short_s]
    dur = max(short[-1, 0] - short[0, 0], 1e-3)
    fs = (len(short) - 1) / dur
    feats = []
    for ci in (1, 2):
        s, L = short[:, ci], long_[:, ci]
        d = np.diff(s)
        feats += [s.mean(), s.std(), np.ptp(s), np.abs(d).mean(), (d != 0).mean()]
        feats += _band_powers(s, fs)
        feats += [L.std(), np.ptp(L), (np.diff(L) != 0).mean(), s.mean() - L.mean()]
    xd = short[:, 1] - short[:, 2]
    if short[:, 1].std() > 0 and short[:, 2].std() > 0:
        xc = float(np.corrcoef(short[:, 1], short[:, 2])[0, 1])
    else:
        xc = 0.0
    dtx = short[-1, 5] - short[0, 5]
    dretry = short[-1, 6] - short[0, 6]
    drx = short[-1, 7] - short[0, 7]
    feats += [xd.std(), xc,
              np.nanmean(short[:, 3]), np.nanstd(short[:, 3]), np.nanstd(short[:, 4]),
              dretry / max(dtx, 1.0), drx / dur]
    return np.nan_to_num(np.asarray(feats, dtype=float))
