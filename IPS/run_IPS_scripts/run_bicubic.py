"""Bicubic IPS runner."""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pointgenIPS_mathematica import IPSPointGeneratorMathematica, add_common_cli_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    args = parser.parse_args()

    # === Geometry ===
    ambient = np.array([2, 2])
    deg3 = np.array(
        [m for m in product(range(4), repeat=3) if sum(m) == 3],
        dtype=np.int64,
    )
    bicubic_exps = np.array(
        [np.concatenate([a, b]) for a in deg3 for b in deg3],
        dtype=np.int64,
    )
    monomials = [bicubic_exps]
    coefficients = [np.random.uniform(-1, 1, len(bicubic_exps)).astype(np.complex128)]
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
