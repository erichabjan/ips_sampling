from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator


DEFAULT_INPUT_DIR = Path("/home/habjan.e/CY_metric/data/ips_mathematica_output/bicubic")
DEFAULT_NUM_REGIONS = 1
DEFAULT_VAL_SPLIT = 0.1
SHUFFLE = False
DEFAULT_SEED = 2025
DEFAULT_VOL_MODULI = None  # None -> ones(len(ambient))

def parse_args():
    p = argparse.ArgumentParser(description="Pack Mathematica IPS outputs into cymetric-compatible dataset.npz + basis.pickle")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Directory containing Mathematica CSV/JSON outputs")
    p.add_argument("--num-regions", type=int, default=DEFAULT_NUM_REGIONS,
                   help="Region count used in Mathematica export filename suffixes")
    p.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                   help="Validation split fraction")
    p.add_argument("--shuffle", action="store_true", default=SHUFFLE,
                   help="Shuffle rows before train/val split")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed (used only if --shuffle)")
    return p.parse_args()


def _load_csv_1d(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _load_csv_2d(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _reconstruct_coefficients_from_metadata(metadata: dict):

    coeffs = []
    for eq_coeffs in metadata["coefficients_realimag"]:
        c = np.array([complex(x["re"], x["im"]) for x in eq_coeffs], dtype=np.complex128)
        coeffs.append(c)
    return coeffs


def _ensure_int_array_nested(x):
    return [np.array(eq, dtype=np.int64) for eq in x]


def _train_val_split_indices(n, val_split=0.1, shuffle=False, seed=0):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    t_i = int((1.0 - val_split) * n)
    return idx[:t_i], idx[t_i:]


def _maybe_make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _get_geometry_from_metadata(metadata: dict):

    if "ambient" in metadata:
        ambient = np.array(metadata["ambient"], dtype=np.int64)
    elif "dimPs" in metadata:
        ambient = np.array(metadata["dimPs"], dtype=np.int64)
    else:
        raise KeyError("metadata is missing both 'ambient' and 'dimPs'")

    dimPs = np.array(metadata.get("dimPs", ambient.tolist()), dtype=np.int64)

    if "exponents" not in metadata:
        raise KeyError("metadata is missing 'exponents'")

    monomials = _ensure_int_array_nested(metadata["exponents"])
    coefficients = _reconstruct_coefficients_from_metadata(metadata)

    return ambient, dimPs, monomials, coefficients


def _load_optional_int_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr)
    return arr.astype(np.int64)


def main():

    args = parse_args()

    INPUT_DIR: Path = args.input_dir
    NUM_REGIONS: int = args.num_regions
    VAL_SPLIT: float = args.val_split
    SHUFFLE: bool = bool(args.shuffle)
    SEED: int = args.seed

    _maybe_make_dir(INPUT_DIR)

    points_real_f = INPUT_DIR / f"points_real_{NUM_REGIONS}.csv"
    points_imag_f = INPUT_DIR / f"points_imag_{NUM_REGIONS}.csv"
    weights_f = INPUT_DIR / f"weights_{NUM_REGIONS}.csv"
    omegas_f = INPUT_DIR / f"omegas_{NUM_REGIONS}.csv"
    kappas_f = INPUT_DIR / f"kappas_{NUM_REGIONS}.csv"
    metadata_f = INPUT_DIR / f"metadata_{NUM_REGIONS}.json"

    patches_local_f = INPUT_DIR / f"patches_local_{NUM_REGIONS}.csv"
    patches_global_f = INPUT_DIR / f"patches_global_{NUM_REGIONS}.csv"
    j_elim_f = INPUT_DIR / f"j_elim_global_{NUM_REGIONS}.csv"

    required = [points_real_f, points_imag_f, weights_f, omegas_f, metadata_f]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    points_real = _load_csv_2d(points_real_f).astype(np.float64)
    points_imag = _load_csv_2d(points_imag_f).astype(np.float64)
    weights = _load_csv_1d(weights_f).astype(np.float64)
    omegas = _load_csv_1d(omegas_f).astype(np.float64)

    if points_real.shape != points_imag.shape:
        raise ValueError(f"points_real shape {points_real.shape} != points_imag shape {points_imag.shape}")

    n_pts, n_coords = points_real.shape

    if len(weights) != n_pts:
        raise ValueError(f"weights length {len(weights)} != number of points {n_pts}")
    if len(omegas) != n_pts:
        raise ValueError(f"omegas length {len(omegas)} != number of points {n_pts}")

    points = points_real + 1j * points_imag

    finite_mask = (
        np.all(np.isfinite(points_real), axis=1)
        & np.all(np.isfinite(points_imag), axis=1)
        & np.isfinite(weights)
        & np.isfinite(omegas)
    )
    pos_mask = (weights > 0) & (omegas >= 0)
    mask = finite_mask & pos_mask

    if not np.all(mask):
        n_bad = np.count_nonzero(~mask)
        print(f"[pack] Filtering {n_bad} invalid rows.")
        points = points[mask]
        points_real = points_real[mask]
        points_imag = points_imag[mask]
        weights = weights[mask]
        omegas = omegas[mask]

    n_pts = len(points)
    print(f"[pack] Using {n_pts} points with {n_coords} coordinates each.")

    metadata = _load_json(metadata_f)
    print(f"[pack] Loaded metadata from {metadata_f.name}")

    kappas = _load_csv_1d(kappas_f).astype(np.float64) if kappas_f.exists() else None
    if kappas is not None:
        print(f"[pack] kappas: shape={kappas.shape}, values={kappas}")

    md_n = metadata.get("num_points_valid", None)
    if md_n is not None and int(md_n) != n_pts:
        print(f"[pack] Warning: metadata num_points_valid={md_n} but CSV-valid rows after filtering={n_pts}")

    md_dim = metadata.get("point_dimension", None)
    if md_dim is not None and int(md_dim) != n_coords:
        print(f"[pack] Warning: metadata point_dimension={md_dim} but CSV point dimension={n_coords}")

    omega_quantity = metadata.get("omega_quantity", None)
    if omega_quantity is not None:
        print(f"[pack] metadata omega_quantity={omega_quantity!r}")
        if omega_quantity != "|Omega|^2":
            print("[pack] Warning: Expected Mathematica omegas to be '|Omega|^2'. Verify convention before training.")

    # Reconstruct point generator
    ambient, dimPs, monomials, coefficients = _get_geometry_from_metadata(metadata)
    if not np.array_equal(ambient, dimPs):
        print("[pack] Warning: ambient != dimPs in metadata (using ambient for CICYPointGenerator).")

    # Kähler moduli
    if "vol_moduli" in metadata:
        vol_moduli = np.array(metadata["vol_moduli"], dtype=np.float64)
    elif DEFAULT_VOL_MODULI is not None:
        vol_moduli = np.array(DEFAULT_VOL_MODULI, dtype=np.float64)
    else:
        vol_moduli = np.ones(len(ambient), dtype=np.float64)

    if len(vol_moduli) != len(ambient):
        raise ValueError(f"len(vol_moduli)={len(vol_moduli)} != len(ambient)={len(ambient)}")

    print("[pack] Instantiating CICYPointGenerator...")
    pg = CICYPointGenerator(monomials, coefficients, vol_moduli, ambient)

    # Train/val split
    train_idx, val_idx = _train_val_split_indices(n_pts, val_split=VAL_SPLIT, shuffle=SHUFFLE, seed=SEED)

    # dataset.npz format
    # X contains concatenated real/imag projective coordinates
    # y[:,0] = IPS weights, y[:,1] = |Omega|^2
    X = np.concatenate([points.real, points.imag], axis=1).astype(np.float64)
    y = np.column_stack([weights, omegas]).astype(np.float64)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Compute val_pullbacks from complex validation points
    points_val = points[val_idx]
    print(f"[pack] Computing val_pullbacks for {len(points_val)} validation points...")
    val_pullbacks = pg.pullbacks(points_val)

    dataset_path = INPUT_DIR / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_pullbacks=val_pullbacks,
    )
    print(f"[pack] Wrote {dataset_path}")

    # Choose kappa for basis
    if kappas is not None and len(kappas) > 0:
        if len(kappas) == 1:
            kappa_for_basis = float(kappas[0])
        else:
            kappa_for_basis = float(np.mean(kappas))
            print(f"[pack] Multiple kappas found; using mean(kappas)={kappa_for_basis:.16e} for basis.")
    else:
        # fallback: estimate with cymetric using loaded data
        print("[pack] No kappas CSV found; estimating kappa via pg.compute_kappa(...)")
        kappa_for_basis = float(pg.compute_kappa(points, weights, omegas))

    print(f"[pack] Writing basis.pickle with kappa={kappa_for_basis:.16e}")
    pg.prepare_basis(str(INPUT_DIR), kappa=kappa_for_basis)  # writes basis.pickle via prepare_basis_pickle
    print(f"[pack] basis.pickle written in {INPUT_DIR}")

    # Save optional diagnostics
    extras = {}
    if patches_local_f.exists():
        pl = _load_optional_int_csv(patches_local_f)
        extras["patches_local_mathematica_1idx"] = pl
        extras["patches_local_python_0idx"] = pl - 1

    if patches_global_f.exists():
        pglo = _load_optional_int_csv(patches_global_f)
        extras["patches_global_mathematica_1idx"] = pglo
        extras["patches_global_python_0idx"] = pglo - 1

    if j_elim_f.exists():
        jel = _load_optional_int_csv(j_elim_f)
        extras["j_elim_global_mathematica_1idx"] = jel
        extras["j_elim_global_python_0idx"] = jel - 1

    if extras:
        extras_path = INPUT_DIR / "mathematica_extras.npz"
        np.savez_compressed(extras_path, **extras)
        print(f"[pack] Wrote optional diagnostics {extras_path}")

    summary = {
        "num_regions": int(NUM_REGIONS),
        "n_points_used": int(n_pts),
        "n_coords": int(n_coords),
        "val_split": float(VAL_SPLIT),
        "shuffle": bool(SHUFFLE),
        "seed": int(SEED),
        "ambient": ambient.tolist(),
        "dimPs": dimPs.tolist(),
        "vol_moduli": vol_moduli.tolist(),
        "omega_quantity_from_metadata": metadata.get("omega_quantity", None),
        "kappa_for_basis": float(kappa_for_basis),
        "dataset_file": "dataset.npz",
        "basis_file": "basis.pickle",
    }
    with open(INPUT_DIR / "pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pack] Wrote {INPUT_DIR / 'pack_summary.json'}")
    print("[pack] Done.")
    print(f"       X_train: {X_train.shape}")
    print(f"       y_train: {y_train.shape}")
    print(f"       X_val:   {X_val.shape}")
    print(f"       y_val:   {y_val.shape}")
    print(f"       val_pullbacks: {val_pullbacks.shape}")


if __name__ == "__main__":
    main()