"""Sample the bicubic using cymetric, with geometry read from an IPS metadata file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointgen_cymetric import sample_and_pack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ips-metadata", type=Path, required=True,
                   help="Path to IPS metadata.json with geometry to mirror.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write cymetric-style outputs.")
    p.add_argument("--num-points", type=int, default=10000)
    p.add_argument("--num-regions", type=int, default=1,
                   help="Kept for CLI compatibility; cymetric baseline is always 1.")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    sample_and_pack(
        ips_metadata_path=args.ips_metadata,
        output_dir=args.output_dir,
        num_points=args.num_points,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
