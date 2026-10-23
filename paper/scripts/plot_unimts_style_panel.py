#!/usr/bin/env python3
"""UniMTS-Fig.4 style 3x6 few-shot panel for TipGuided paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

IN = Path("/hy-tmp/logs/paper_figs/panel_data.json")
OUT = Path("/hy-tmp/logs/paper_figs")


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


# muted baselines + bold red hero (match UniMTS Fig.4 emphasis)
STYLES = {
    "ZS": dict(color="#9AA0A6", marker="x", lw=1.1, ms=4.5, z=2),
    "Probe": dict(color="#C4A484", marker="P", lw=1.1, ms=4.5, z=2),
    "1-NN": dict(color="#7EB6D9", marker="o", lw=1.15, ms=4.5, z=3),
    "Tip-F": dict(color="#E8A87C", marker="s", lw=1.2, ms=4.2, z=3),
    "Full FT": dict(color="#6B8F71", marker="D", lw=1.25, ms=4.0, z=4),
    "TipGuided": dict(color="#E53935", marker="*", lw=2.15, ms=8.5, z=6),
}


def series(data, method, ds, metric="f1"):
    ks = data["ks"]
    ys = []
    for k in ks:
        cell = data["data"][method][str(k) if isinstance(k, str) else k].get(ds)
        # json may stringify keys when reloaded - handle both
        if cell is None:
            # try string k
            cell = data["data"][method].get(str(k), {}).get(ds)
        if cell is None:
            ys.append(np.nan)
        else:
            ys.append(100.0 * float(cell[metric]))
    return np.array(ys, dtype=float)


def fix_keys(data):
    """Ensure nested k keys are ints."""
    new = {}
    for m, byk in data["data"].items():
        new[m] = {}
        for k, v in byk.items():
            new[m][int(k)] = v
    data["data"] = new
    data["ks"] = [int(k) for k in data["ks"]]
    return data


def plot_panel(metric="f1", fname="fig_fewshot_panel_f1"):
    data = fix_keys(json.loads(IN.read_text()))
    datasets = data["datasets"]
    display = data["display"]
    ks = data["ks"]
    methods = ["ZS", "Probe", "1-NN", "Tip-F", "Full FT", "TipGuided"]

    fig, axes = plt.subplots(3, 6, figsize=(12.8, 6.6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for m in methods:
            y = series(data, m, ds, metric=metric)
            st = STYLES[m]
            ax.plot(
                ks,
                y,
                color=st["color"],
                marker=st["marker"],
                linewidth=st["lw"],
                markersize=st["ms"],
                markerfacecolor=st["color"],
                markeredgecolor="white" if m == "TipGuided" else st["color"],
                markeredgewidth=0.6 if m == "TipGuided" else 0.4,
                label=m,
                zorder=st["z"],
                solid_capstyle="round",
            )
        ax.set_title(display.get(ds, ds), pad=3)
        ax.set_xticks(ks)
        ax.set_xlim(0.5, 10.8)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, which="both", color="#E5E7EB", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#B0B5BB")
            spine.set_linewidth(0.8)

        # left y label per row
        if i % 6 == 0:
            ax.set_ylabel("Macro-F1" if metric == "f1" else "Accuracy (%)")
        # bottom x label
        if i >= 12:
            ax.set_xlabel("Number of Samples")

    # shared legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.08, wspace=0.12, hspace=0.28)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)
    print("wrote", OUT / fname)


def plot_mean_curve(metric="f1", fname="fig_fewshot_mean_f1"):
    """Companion mean± over datasets (not runs) as light band — honest: dataset variation."""
    data = fix_keys(json.loads(IN.read_text()))
    ks = data["ks"]
    methods = ["ZS", "Probe", "1-NN", "Tip-F", "Full FT", "TipGuided"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in methods:
        ys = []
        for k in ks:
            vals = [
                100.0 * data["data"][m][k][ds][metric]
                for ds in data["datasets"]
                if ds in data["data"][m][k] and data["data"][m][k][ds].get(metric) is not None
            ]
            ys.append(vals)
        mean = np.array([np.mean(v) for v in ys])
        std = np.array([np.std(v) for v in ys])  # across datasets, disclosed in caption
        st = STYLES[m]
        ax.plot(
            ks,
            mean,
            color=st["color"],
            marker=st["marker"],
            lw=st["lw"] + (0.4 if m == "TipGuided" else 0),
            ms=st["ms"],
            label=m,
            zorder=st["z"],
            markeredgecolor="white" if m == "TipGuided" else st["color"],
        )
        ax.fill_between(ks, mean - std, mean + std, color=st["color"], alpha=0.12 if m != "TipGuided" else 0.18, zorder=1)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Macro-F1" if metric == "f1" else "Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7)
    ax.set_title("18-dataset mean Macro-F1 (band = std across datasets)")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


def main():
    style()
    plot_panel("f1", "fig_fewshot_panel_f1")
    plot_panel("acc", "fig_fewshot_panel_acc")
    plot_mean_curve("f1", "fig_fewshot_mean_f1")
    plot_mean_curve("acc", "fig_fewshot_mean_acc")


if __name__ == "__main__":
    main()
