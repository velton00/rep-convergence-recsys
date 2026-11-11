"""Alignment figure for the paper and the slides."""

import logging
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

VISION_LABELS = {
    "vit_small_patch14_dinov2.lvd142m": "DINOv2-S",
    "vit_base_patch14_dinov2.lvd142m": "DINOv2-B",
    "vit_large_patch14_dinov2.lvd142m": "DINOv2-L",
    "vit_giant_patch14_dinov2.lvd142m": "DINOv2-G",
}
# one hue, light to dark with model size; checked as an ordinal ramp on white
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

STYLES = {
    "slides": {
        "font": "CMU Sans Serif",
        "size": 8,
        "figsize": (2.9, 2.3),
        "margins": {"left": 0.18, "right": 0.78, "top": 0.97, "bottom": 0.19},
        "xlabel": "BLOOMZ, млрд параметров",
        "label_gap": 0.0088,
    },
    "article": {
        "font": "CMU Serif",
        "size": 9.5,
        "figsize": (4.6, 3.0),
        "margins": {"left": 0.13, "right": 0.83, "top": 0.97, "bottom": 0.15},
        "xlabel": "Размер языковой модели BLOOMZ, млрд параметров",
        "label_gap": 0.0072,
    },
}
CMU_FONTS = Path("/usr/share/fonts/truetype/cmu")

# CMU regular faces are registered with weight 500, which matplotlib reports on every call;
# fontTools does the same for every font it subsets into the PDF
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)


def _font(name: str) -> str:
    """Register Computer Modern Unicode if it is installed, otherwise fall back to the default."""
    if CMU_FONTS.is_dir():
        for path in CMU_FONTS.glob("cmun*.ttf"):
            font_manager.fontManager.addfont(str(path))
    available = {f.name for f in font_manager.fontManager.ttflist}
    return name if name in available else "DejaVu Sans"


def _comma(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _spread(values: list[float], gap: float) -> list[float]:
    """Move sorted label positions apart until neighbours are at least ``gap`` away."""
    pos = list(values)
    for _ in range(50):
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                mid = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = mid - gap / 2, mid + gap / 2
    return pos


def plot_alignment(
    results: pd.DataFrame,
    path: Path,
    style: str = "article",
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Alignment against language model size, one line per vision model.

    Writes ``path`` once per format, e.g. ``alignment.pdf`` and ``alignment.png``.
    """
    st = STYLES[style]
    scale = st["size"] / 8
    rc = {
        "font.family": _font(st["font"]),
        "font.size": st["size"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=st["figsize"])
        fig.subplots_adjust(**st["margins"])
        ax = fig.subplots()

        vision = results.groupby("lvm")["lvm_params_m"].first().sort_values().index
        ends = []
        for name, color in zip(vision, BLUES, strict=False):
            d = results[results["lvm"] == name].sort_values("llm_params_b")
            label = VISION_LABELS.get(name, name)
            ax.plot(
                d["llm_params_b"],
                d["score"],
                color=color,
                lw=1.4,
                marker="o",
                ms=4.2,
                mec="white",
                mew=0.8,
                label=label,
                zorder=3,
            )
            ends.append((float(d["score"].iloc[-1]), float(d["llm_params_b"].iloc[-1]), label))

        ends.sort()
        for (y, x, label), y_text in zip(
            ends, _spread([e[0] for e in ends], st["label_gap"]), strict=True
        ):
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x * 1.13, y_text),
                fontsize=7.5 * scale,
                color=INK_SOFT,
                va="center",
                annotation_clip=False,
            )

        sizes = sorted(results["llm_params_b"].unique())
        chance = float(results["chance"].iloc[0])
        ax.axhline(chance, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(
            sizes[0] / 1.19,
            chance + 0.0035,
            "случайный уровень",
            fontsize=7 * scale,
            color=INK_SOFT,
            va="bottom",
        )

        ax.set_xscale("log")
        ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.27)
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}".replace(".", ",")))
        ax.set_ylim(0, max(0.145, results["score"].max() * 1.1))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.04, 0.08, 0.12]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _comma(v)))

        ax.set_xlabel(st["xlabel"], color=INK, fontsize=8 * scale)
        ax.set_ylabel(r"Выравнивание, mutual $k$-NN", color=INK, fontsize=8 * scale)
        ax.tick_params(colors=INK_SOFT, labelsize=7.5 * scale, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)

        legend = ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 0.12),
            ncol=2,
            frameon=False,
            fontsize=7 * scale,
            handlelength=1.6,
            columnspacing=1.0,
            handletextpad=0.4,
            title="Модель зрения",
            title_fontsize=7 * scale,
        )
        legend.get_title().set_color(INK_SOFT)
        for text in legend.get_texts():
            text.set_color(INK_SOFT)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(path.with_suffix(f".{fmt}"), dpi=300)
