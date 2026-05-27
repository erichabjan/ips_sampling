"""Quintic-throat IPS runner."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pointgenIPS_mathematica import IPSPointGeneratorMathematica, add_common_cli_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    args = parser.parse_args()

    # === Geometry ===
    ambient = np.array([4])
    monomials = [np.array([
        [5, 0, 0, 0, 0],
        [0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 5],
        [1, 1, 1, 1, 1],
    ], dtype=np.int64)]
    psi = 1.035 - 0.02j
    coefficients = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, -5.0 * psi], dtype=np.complex128)]
    kmoduli = np.ones(len(ambient))
    target_volume = None

    pg = IPSPointGeneratorMathematica(
        monomials, coefficients, kmoduli, ambient,
        precision=args.precision,
        verbose=args.verbose,
        vol_j_norm=target_volume,
        kernel_path=args.kernel_path,
    )

    pg.prepare_dataset(
        args.output_dir,
        total_pts=args.total_points,
        num_regions=args.num_regions,
        val_split=args.val_split,
        shuffle=args.shuffle,
        seed=args.seed,
        nproc=args.nproc,
        front_end=False,
    )


if __name__ == "__main__":
    main()
