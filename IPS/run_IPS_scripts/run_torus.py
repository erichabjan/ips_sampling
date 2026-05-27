"""Weierstrass cubic (torus) IPS runner."""

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
    ambient = np.array([2])
    monomials = [np.array([[1, 0, 2], [0, 3, 0], [2, 1, 0]], dtype=np.int64)]
    coefficients = [np.array([1.0, -4.0, 189.07272], dtype=np.complex128)]
    kmoduli = np.ones(len(ambient))
    target_volume = None  # was Automatic

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
