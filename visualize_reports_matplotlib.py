#!/usr/bin/env python3
"""
Visualize speech-enhancement report CSVs as boxplots + jittered points (matplotlib only).

Expected filenames (best-effort parser):
  report.<split>.<device>.<signal_type>.dev_02.P091.csv
Example:
  report.dev.ha.individual.dev_02.P091.csv

Usage examples:
  python visualize_reports_matplotlib.py \
    --reports_dir "/no_backups/s1495/experiments/baseline_1/evaluation/reports" \
    --metrics pysepm_fwsegsnr sdr \
    --out "metrics.png"

  # If you only want one metric
  python visualize_reports_matplotlib.py --reports_dir ./reports --metrics pysepm_fwsegsnr
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def parse_report_filename(p: Path) -> dict:
    parts = p.stem.split(".")
    meta = {"split": None, "device": None, "signal_type": None, "session": None, "pid": None, "file": str(p)}
    if len(parts) >= 5 and parts[0] == "report":
        meta["split"] = parts[1]
        meta["device"] = parts[2]
        meta["signal_type"] = parts[3]
        meta["session"] = parts[4]
        meta["pid"] = parts[5] if len(parts) >= 6 else None
    else:
        m = re.search(r"(dev_\d+)", p.stem)
        meta["session"] = m.group(1) if m else p.stem
    return meta


def sort_sessions(sessions: list[str]) -> list[str]:
    def key(s: str):
        m = re.search(r"dev_(\d+)", s)
        return (0, int(m.group(1))) if m else (1, s)
    return sorted(sessions, key=key)


def box_scatter(ax, values_by_group, labels, seed=0, delta=0.22, jitter=0.12):
    centers = np.arange(1, len(labels) + 1)

    box_pos = centers + delta        # box a bit to the right
    pts_pos = centers - delta        # points a bit to the left

    bp = ax.boxplot(
        values_by_group,
        positions=box_pos,
        widths=0.38,
        showfliers=False, #shows outliers as points(fliers)
        patch_artist=True,
        manage_ticks=False,          # we set ticks ourselves
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i in range(len(labels)):
        c = colors[i % len(colors)] if colors else None
        bp["boxes"][i].set_facecolor("none")
        if c is not None:
            bp["boxes"][i].set_edgecolor(c)
            bp["medians"][i].set_color(c)
            bp["whiskers"][2*i].set_color(c); bp["whiskers"][2*i+1].set_color(c)
            bp["caps"][2*i].set_color(c);     bp["caps"][2*i+1].set_color(c)

    rng = np.random.default_rng(seed)
    for i, vals in enumerate(values_by_group):
        if len(vals) == 0:
            continue
        x = pts_pos[i] + rng.uniform(-jitter, jitter, size=len(vals))
        c = colors[i % len(colors)] if colors else None
        ax.scatter(x, vals, s=10, alpha=0.45, c=c)

    # --- Legend (one entry per session) ---
    handles = []
    for i, lab in enumerate(labels):
        c = colors[i % len(colors)] if colors else "C0"
        handles.append(Patch(facecolor=c, edgecolor=c, label=lab))

    ax.legend(
        handles=handles,
        title="Session",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(0, linewidth=1.0)
    ax.set_xlim(0.5, len(labels) + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", type=str, required=True, help="Directory containing report.*.csv files")
    ap.add_argument("--pattern", type=str, default="report.*.csv", help="Glob pattern (default: report.*.csv)")
    ap.add_argument("--metrics", nargs="+", required=True, help="Column names to plot (stacked as rows)")
    ap.add_argument("--filter_split", type=str, default=None, help="Keep only this split (e.g., dev)")
    ap.add_argument("--filter_device", type=str, default=None, help="Keep only this device (e.g., ha)")
    ap.add_argument("--filter_signal_type", type=str, default=None, help="Keep only this signal_type (e.g., individual)")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path (default: <reports_dir>/metrics.png)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    paths = sorted(reports_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {reports_dir / args.pattern}")

    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        meta = parse_report_filename(p)
        for k, v in meta.items():
            d[k] = v
        dfs.append(d)
    data = pd.concat(dfs, ignore_index=True)

    if args.filter_split is not None:
        data = data[data["split"] == args.filter_split]
    if args.filter_device is not None:
        data = data[data["device"] == args.filter_device]
    if args.filter_signal_type is not None:
        data = data[data["signal_type"] == args.filter_signal_type]

    present_signal_types = sorted(str(x) for x in data["signal_type"].dropna().unique())
    signal_type_label = ", ".join(present_signal_types) if present_signal_types else "unknown"

    sessions = sort_sessions([s for s in data["session"].dropna().unique()])
    if not sessions:
        raise ValueError("No sessions found after filtering.")

    # Validate metrics
    missing = [m for m in args.metrics if m not in data.columns]
    if missing:
        raise KeyError(f"Metrics not found in CSV columns: {missing}")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        nrows=len(args.metrics),
        ncols=1,
        figsize=(max(10, 1.2 * len(sessions)), 3.2 * len(args.metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(args.metrics) == 1:
        axes = [axes]
    fig.suptitle(f"signal_type: {signal_type_label}", fontsize=12, y=1.02)
    for ax, metric in zip(axes, args.metrics):
        values = [data.loc[data["session"] == s, metric].dropna().to_numpy() for s in sessions]
        box_scatter(ax, values, sessions, seed=args.seed)
        ax.set_ylabel(metric)

    axes[-1].set_xlabel("Session")

    out = Path(args.out) if args.out else (reports_dir / "metrics.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
