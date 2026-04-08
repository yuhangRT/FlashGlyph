#!/usr/bin/env python3
"""Plot effectiveness figures for FlashGlyph.

Usage:
  python student_model_v3/experiments/plot_effectiveness.py \
      --out_dir student_model_v3/experiments/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Pareto frontier for minimizing x and maximizing y."""
    work = df.sort_values(x_col, ascending=True).reset_index(drop=True)
    keep = []
    best_y = float("-inf")
    for i, row in work.iterrows():
        y = float(row[y_col])
        if y > best_y:
            keep.append(i)
            best_y = y
    return work.loc[keep].copy()


def classify_method(name: str) -> str:
    if "FlashGlyph" in name:
        return "flashglyph"
    if "LCM-baseline" in name:
        return "lcm"
    if "AnyText2" in name:
        return "teacher"
    return "solver"


STYLE = {
    "flashglyph": dict(color="#C43C39", marker="o", s=130, alpha=0.98, label="FlashGlyph"),
    "lcm": dict(color="#F28E2B", marker="s", s=96, alpha=0.95, label="LCM baseline"),
    "teacher": dict(color="#4E79A7", marker="*", s=300, alpha=0.98, label="AnyText2 teacher"),
    "solver": dict(color="#A7A9AC", marker="D", s=62, alpha=0.82, label="Teacher + solver"),
}


def set_plot_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "#4c4c4c",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def _latency_tick_formatter(x, _):
    if x >= 1000:
        if abs(x % 1000) < 1e-6:
            return f"{int(x/1000)}k"
        return f"{x/1000:.1f}k"
    return f"{int(x)}"


def _label_offsets(panel: str):
    if panel == "EN":
        return {
            "FlashGlyph (ours)": (10, 8),
            "LCM-baseline (mask)": (10, 6),
            "AnyText2 (Teacher)": (8, 5),
        }
    return {
        "FlashGlyph (ours)": (10, 8),
        "LCM-baseline (mask)": (10, 6),
        "AnyText2 (Teacher)": (8, 5),
    }


def _draw_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    for fam in ["solver", "lcm", "flashglyph", "teacher"]:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        st = STYLE[fam]
        edgecolor = "white" if fam in {"flashglyph", "teacher", "lcm"} else "none"
        linewidth = 1.0 if fam in {"flashglyph", "teacher", "lcm"} else 0.0
        ax.scatter(
            sub["latency_ms"],
            sub["char_acc"],
            c=st["color"],
            marker=st["marker"],
            s=st["s"],
            alpha=st["alpha"],
            label=st["label"],
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=4 if fam in {"flashglyph", "teacher"} else 3,
        )


def _draw_frontier(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    frontier = pareto_frontier(df, "latency_ms", "char_acc")
    ax.plot(
        frontier["latency_ms"],
        frontier["char_acc"],
        linestyle=(0, (4, 2)),
        color="#2F9E44",
        linewidth=2.2,
        label="Pareto frontier",
        zorder=2,
    )
    return frontier


def _add_focus_inset(ax: plt.Axes, df: pd.DataFrame) -> None:
    zoom = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1.1)
    focus = df[df["latency_ms"] <= 1600].copy()
    zoom.set_facecolor("#FCFCFC")
    _draw_scatter(zoom, focus)
    _draw_frontier(zoom, focus)
    zoom.set_xlim(180, 1550)
    zoom.set_ylim(focus["char_acc"].min() - 1.5, focus["char_acc"].max() + 1.8)
    zoom.set_xticks([250, 500, 1000, 1500])
    zoom.set_yticks([75, 80, 85, 90])
    zoom.grid(True, which="major", linestyle=":", alpha=0.55)
    zoom.tick_params(axis="both", labelsize=8, pad=1)
    zoom.set_title("Low-latency region", fontsize=8.8, pad=2.5)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#777777")


def plot_main_effectiveness(
    df: pd.DataFrame,
    title: str,
    panel: str,
    ax: plt.Axes,
    show_xlabel: bool,
) -> None:
    df = df.copy()
    df["family"] = df["method"].map(classify_method)

    _draw_scatter(ax, df)
    _draw_frontier(ax, df)

    offsets = _label_offsets(panel)
    for key in ["FlashGlyph (ours)", "LCM-baseline (mask)", "AnyText2 (Teacher)"]:
        row = df[df["method"] == key]
        if row.empty:
            continue
        x = float(row.iloc[0]["latency_ms"])
        y = float(row.iloc[0]["char_acc"])
        label = key.replace(" (ours)", "").replace(" (Teacher)", "")
        dx, dy = offsets.get(key, (6, 4))
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            fontweight="semibold" if "FlashGlyph" in key else "normal",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#D8D8D8",
                linewidth=0.6,
                alpha=0.96,
            ),
            zorder=5,
        )

    _add_focus_inset(ax, df)

    ax.set_xscale("log")
    ax.set_xlim(180, 13000)
    ax.set_ylim(56, 97)
    if show_xlabel:
        ax.set_xlabel("Latency (ms, lower is better)")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Char Acc (%)")
    ax.set_title(title, loc="left", pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(_latency_tick_formatter))
    ax.grid(True, which="major", linestyle=":", alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", alpha=0.16)
    ax.set_axisbelow(True)


def plot_ablation(df_cn: pd.DataFrame, df_en: pd.DataFrame, out_path: Path) -> None:
    order = list(df_cn["variant"])
    short = ["Baseline", "+Attn", "+Attn+OCR", "+Topo", "+Sharp"]
    x = range(len(order))

    fig, axes = plt.subplots(2, 1, figsize=(7.3, 8.8), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(
        x,
        df_cn["char_acc"],
        marker="o",
        color=STYLE["flashglyph"]["color"],
        linewidth=3.0,
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    axes[0].set_xticks(list(x), short, rotation=0)
    axes[0].set_title("(a) Chinese Ablation", loc="left", pad=8)
    axes[0].set_ylabel("Char Acc (%)")
    axes[0].set_ylim(df_cn["char_acc"].min() - 0.4, df_cn["char_acc"].max() + 0.4)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].set_axisbelow(True)

    axes[1].plot(
        x,
        df_en["char_acc"],
        marker="o",
        color=STYLE["teacher"]["color"],
        linewidth=3.0,
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    axes[1].set_xticks(list(x), short, rotation=0)
    axes[1].set_title("(b) English Ablation", loc="left", pad=8)
    axes[1].set_ylabel("Char Acc (%)")
    axes[1].set_xlabel("Model Variant")
    axes[1].set_ylim(df_en["char_acc"].min() - 0.4, df_en["char_acc"].max() + 0.4)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].set_axisbelow(True)

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=0)

    fig.subplots_adjust(hspace=0.28, bottom=0.11, top=0.96)
    fig.savefig(out_path, dpi=320)
    fig.savefig(out_path.with_suffix(".svg"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("student_model_v3/experiments/predicted"),
        help="Directory that contains table1a/1b/table2 csv files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("student_model_v3/experiments/figures"),
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    set_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t1a = pd.read_csv(args.pred_dir / "table1a_cn_predicted.csv")
    t1b = pd.read_csv(args.pred_dir / "table1b_en_predicted.csv")
    t2a = pd.read_csv(args.pred_dir / "table2_cn_predicted.csv")
    t2b = pd.read_csv(args.pred_dir / "table2_en_predicted.csv")

    fig, axes = plt.subplots(2, 1, figsize=(7.3, 9.2), sharex=True)
    fig.patch.set_facecolor("white")
    plot_main_effectiveness(t1a, "(a) Main Effectiveness on Chinese Text", "CN", axes[0], show_xlabel=False)
    plot_main_effectiveness(t1b, "(b) Main Effectiveness on English Text", "EN", axes[1], show_xlabel=True)

    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=STYLE["solver"]["color"], markersize=7, label="Teacher + solver"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=STYLE["lcm"]["color"], markeredgecolor="white", markersize=8, label="LCM baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STYLE["flashglyph"]["color"], markeredgecolor="white", markersize=8.5, label="FlashGlyph"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=STYLE["teacher"]["color"], markeredgecolor="white", markersize=12.5, label="AnyText2 teacher"),
        Line2D([0], [0], linestyle=(0, (4, 2)), color="#2F9E44", linewidth=2.2, label="Pareto frontier"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=9.8,
        columnspacing=1.15,
        handletextpad=0.7,
    )
    fig.subplots_adjust(hspace=0.24, bottom=0.12, top=0.96)

    main_out = args.out_dir / "fig_effectiveness_pareto.png"
    fig.savefig(main_out, dpi=340)
    fig.savefig(main_out.with_suffix(".svg"))
    fig.savefig(main_out.with_suffix(".pdf"))
    plt.close(fig)

    ablation_out = args.out_dir / "fig_effectiveness_ablation.png"
    plot_ablation(t2a, t2b, ablation_out)

    print(f"[OK] saved: {main_out}")
    print(f"[OK] saved: {ablation_out}")


if __name__ == "__main__":
    main()
