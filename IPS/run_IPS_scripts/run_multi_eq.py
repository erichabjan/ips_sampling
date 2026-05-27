"""Multi-equation CICY IPS runner (P^3 x P^3 with three equations)."""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pointgenIPS_mathematica import IPSPointGeneratorMathematica, add_common_cli_args


def _block_exponents(d, n):
    return np.array(
        [m for m in product(range(d + 1), repeat=n) if sum(m) == d],
        dtype=np.int64,
    )


def _equation_exponents(deg_row, ambient):
    blocks = [_block_exponents(deg_row[i], ambient[i] + 1) for i in range(len(ambient))]
    return np.array(
        [np.concatenate(combo) for combo in product(*blocks)],
        dtype=np.int64,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    args = parser.parse_args()

    # === Geometry ===
    ambient = np.array([3, 3])
    degree_matrix = np.array([[3, 0], [0, 3], [1, 1]], dtype=np.int64)
    monomials = [_equation_exponents(row, ambient) for row in degree_matrix]
    coefficients = [
        np.random.uniform(-1, 1, len(m)).astype(np.complex128) for m in monomials
    ]
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
