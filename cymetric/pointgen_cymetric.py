"""Cymetric baseline sampler that mirrors the IPS workhorse output layout.

Reads a CICY geometry from an IPS-produced ``metadata.json``, samples points
with cymetric's ``CICYPointGenerator``, and writes the same set of files the
IPS workhorse writes (raw fixed-name CSV/JSON + cymetric-standard
``dataset.npz`` / ``basis.pickle`` / ``point_gen.pickle`` /
``mathematica_extras.npz`` / ``pack_summary.json``).

Single entry point: :func:`sample_and_pack`. The per-CICY ``*_sampling.py``
scripts are thin wrappers around it.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Union

import numpy as np

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator


logger = logging.getLogger("pointgen_cymetric")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(name)s:%(levelname)s:%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _patch_globals_from_local(patch_local: np.ndarray, ambient: np.ndarray) -> np.ndarray:
    """1-indexed local patch indices → 1-indexed global flattened indices."""
    ambient = np.asarray(ambient, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(ambient + 1)[:-1]])
    return patch_local + offsets[None, :]


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr[:, None] if arr.ndim == 1 else arr


def _read_ips_metadata(path: Path):
    with open(path) as f:
        md = json.load(f)
    monomials = [np.array(eq, dtype=np.int64) for eq in md["exponents"]]
    coefficients = [
        np.array([complex(x["re"], x["im"]) for x in eq_c], dtype=np.complex128)
        for eq_c in md["coefficients_realimag"]
    ]
    ambient = np.array(md["ambient"], dtype=np.int64)
    kahler_moduli = np.array(
        md.get("kahler_moduli", np.ones(len(ambient)).tolist()),
        dtype=np.float64,
    )
    dim_cy = int(md.get("dim_cy", int(np.sum(ambient) - len(monomials))))
    return monomials, coefficients, ambient, kahler_moduli, dim_cy


def _compute_patches(points: np.ndarray, ambient: np.ndarray):
    """Return (patches_local, patches_global) as ``(n, n_factors)`` 1-indexed arrays."""
    n = len(points)
    n_factors = len(ambient)
    coord_starts = np.concatenate([[0], np.cumsum(ambient + 1)])

    patches_local = np.zeros((n, n_factors), dtype=np.int64)
    patches_global = np.zeros((n, n_factors), dtype=np.int64)
    for k in range(n_factors):
        start, end = coord_starts[k], coord_starts[k + 1]
        local_idx = np.argmax(np.abs(points[:, start:end]), axis=1)
        patches_local[:, k] = local_idx + 1
        patches_global[:, k] = start + local_idx + 1
    return patches_local, patches_global


def _complex_to_json(z):
    return {"re": float(z.real), "im": float(z.imag)}


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def sample_and_pack(
    ips_metadata_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_points: int,
    *,
    val_split: float = 0.1,
    shuffle: bool = False,
    seed: int = 2025,
    oversample_factor: float = 1.05,
) -> float:
    """Sample with cymetric and write the full output layout.

    Returns the kappa value used for the basis.
    """
    ips_metadata_path = Path(ips_metadata_path)
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    monomials, coefficients, ambient, kahler_moduli, dim_cy = _read_ips_metadata(
        ips_metadata_path
    )
    logger.info(
        f"Geometry: ambient={ambient.tolist()}, dim_cy={dim_cy}, "
        f"num_hypersurfaces={len(monomials)}"
    )

    vol_moduli = np.ones(len(ambient), dtype=np.float64)
    pg = CICYPointGenerator(monomials, coefficients, vol_moduli, ambient)

    n_request = int(np.ceil(num_points * oversample_factor))
    logger.info(f"Sampling {n_request} candidate points via cymetric...")
    pwo = pg.generate_point_weights(n_request, omega=True)

    points = pwo["point"]
    weights = pwo["weight"].astype(np.float64)
    omega_complex = pwo["omega"]
    omegas = np.real(omega_complex * np.conj(omega_complex)).astype(np.float64)

    n = min(len(points), num_points)
    points = points[:n].astype(np.complex128)
    weights = weights[:n]
    omegas = omegas[:n]

    # Drop invalid rows
    mask = (
        np.all(np.isfinite(points.real), axis=1)
        & np.all(np.isfinite(points.imag), axis=1)
        & np.isfinite(weights)
        & np.isfinite(omegas)
        & (weights > 0)
        & (omegas >= 0)
    )
    if not np.all(mask):
        n_bad = int(np.count_nonzero(~mask))
        logger.warning(f"Filtering {n_bad} invalid rows.")
        points = points[mask]
        weights = weights[mask]
        omegas = omegas[mask]
    n = len(points)
    logger.info(f"Kept {n} valid points.")

    kappa = float(pg.compute_kappa(points, weights, omegas))
    logger.info(f"kappa = {kappa:.6e}")

    # j_elim and patches
    j_elim_0idx = pg._find_max_dQ_coords(points)
    j_elim_1idx = _ensure_2d(np.asarray(j_elim_0idx, dtype=np.int64) + 1)
    patches_local, patches_global = _compute_patches(points, ambient)
    region_labels = np.ones(n, dtype=np.int64)  # single-region cymetric baseline

    acceptances = np.array([n], dtype=np.float64)
    num_samples = np.array([n_request], dtype=np.float64)
    kappas_arr = np.array([kappa], dtype=np.float64)

    # --- Raw fixed-name CSV/JSON outputs --------------------------------------
    np.savetxt(d / "points_real.csv", points.real, delimiter=",")
    np.savetxt(d / "points_imag.csv", points.imag, delimiter=",")
    np.savetxt(d / "weights.csv", weights, delimiter=",")
    np.savetxt(d / "omegas.csv", omegas, delimiter=",")
    np.savetxt(d / "kappas.csv", kappas_arr, delimiter=",")
    np.savetxt(d / "patches_local.csv", patches_local, delimiter=",", fmt="%d")
    np.savetxt(d / "patches_global.csv", patches_global, delimiter=",", fmt="%d")
    np.savetxt(d / "j_elim_global.csv", j_elim_1idx, delimiter=",", fmt="%d")
    np.savetxt(d / "region_labels.csv", region_labels, delimiter=",", fmt="%d")
    np.savetxt(d / "acceptances.csv", acceptances, delimiter=",")
    np.savetxt(d / "num_samples.csv", num_samples, delimiter=",")

    ncoords = int(np.sum(ambient + 1))
    metadata = {
        "schema_version": 3,
        "generator": "cymetric_CICYPointGenerator",
        "requested_total_points": int(num_points),
        "num_regions_requested": 1,
        "precision": 16,
        "verbose": 1,
        "front_end": False,
        "num_points_valid": n,
        "point_dimension": ncoords,
        "dim_cy": dim_cy,
        "num_hypersurfaces": len(monomials),
        "num_ambient_factors": len(ambient),
        "dimPs": ambient.tolist(),
        "ambient": ambient.tolist(),
        "exponents": [m.tolist() for m in monomials],
        "coefficients_realimag": [
            [_complex_to_json(c) for c in eq_c] for eq_c in coefficients
        ],
        "kahler_moduli": kahler_moduli.tolist(),
        "target_volume": None,
        "omega_quantity": "|Omega|^2",
        "omega_description": "cymetric omegas are |Omega|^2 = real(Omega * conj(Omega))",
        "weights_quantity": "|Omega|^2 / det(g_FS_norm) from cymetric point_weight",
        "patches_local_convention": "1-indexed patch index within each projective block",
        "patches_global_convention": "1-indexed flattened global coordinate indices",
        "j_elim_global_convention": "1-indexed flattened global eliminated coordinate indices",
        "region_labels_convention": "1-indexed metric/region label (all 1 for cymetric)",
        "files": {
            "points_real_csv": "points_real.csv",
            "points_imag_csv": "points_imag.csv",
            "weights_csv": "weights.csv",
            "omegas_csv": "omegas.csv",
            "kappas_csv": "kappas.csv",
            "patches_local_csv": "patches_local.csv",
            "patches_global_csv": "patches_global.csv",
            "j_elim_global_csv": "j_elim_global.csv",
            "region_labels_csv": "region_labels.csv",
            "acceptances_csv": "acceptances.csv",
            "num_samples_csv": "num_samples.csv",
        },
    }
    with open(d / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote raw cymetric outputs to {d}")

    # --- Cymetric-standard outputs --------------------------------------------
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    t_i = int((1.0 - val_split) * n)
    train_idx, val_idx = idx[:t_i], idx[t_i:]

    X = np.concatenate([points.real, points.imag], axis=1).astype(np.float64)
    y = np.column_stack([weights, omegas]).astype(np.float64)

    logger.info(f"Computing val_pullbacks for {len(val_idx)} validation points...")
    val_pullbacks = pg.pullbacks(points[val_idx])

    np.savez_compressed(
        d / "dataset.npz",
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        val_pullbacks=val_pullbacks,
    )
    logger.info(f"Wrote {d/'dataset.npz'}")

    pg.prepare_basis(str(d), kappa=kappa)
    logger.info(f"Wrote basis.pickle (kappa={kappa:.6e})")

    with open(d / "point_gen.pickle", "wb") as f:
        pickle.dump(pg, f)
    logger.info(f"Wrote {d/'point_gen.pickle'}")

    order = np.concatenate([train_idx, val_idx])
    extras = {
        "patches_local_mathematica_1idx": patches_local[order],
        "patches_local_python_0idx": patches_local[order] - 1,
        "patches_global_mathematica_1idx": patches_global[order],
        "patches_global_python_0idx": patches_global[order] - 1,
        "j_elim_global_mathematica_1idx": j_elim_1idx[order],
        "j_elim_global_python_0idx": j_elim_1idx[order] - 1,
        "region_labels_mathematica_1idx": region_labels[order],
        "region_labels_python_0idx": region_labels[order] - 1,
        "acceptances": acceptances,
        "num_samples": num_samples,
    }
    np.savez_compressed(d / "mathematica_extras.npz", **extras)
    logger.info(f"Wrote {d/'mathematica_extras.npz'}")

    summary = {
        "num_regions": 1,
        "n_points_used": int(n),
        "n_coords": int(ncoords),
        "val_split": float(val_split),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "ambient": ambient.tolist(),
        "dimPs": ambient.tolist(),
        "vol_moduli": vol_moduli.tolist(),
        "kahler_moduli": kahler_moduli.tolist(),
        "omega_quantity_from_metadata": "|Omega|^2",
        "kappa_for_basis": float(kappa),
        "dataset_file": "dataset.npz",
        "basis_file": "basis.pickle",
        "has_region_labels": True,
        "has_acceptances": True,
        "has_num_samples": True,
        "has_h_blocks": False,
    }
    with open(d / "pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {d/'pack_summary.json'}")

    return kappa
