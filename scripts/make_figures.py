"""Redraw the alignment figure for the article and the slides from a results table.

python scripts/make_figures.py results/wit1024/alignment.csv tex/figures
"""

import argparse
from pathlib import Path

import pandas as pd

from repconv.plotting import STYLES, plot_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Alignment figures for the article and slides")
    parser.add_argument("table", type=Path, help="alignment.csv written by run_wit1024.py")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    for style in STYLES:
        plot_alignment(table, args.out_dir / f"wit1024_alignment_{style}", style, formats=("pdf",))


if __name__ == "__main__":
    main()
