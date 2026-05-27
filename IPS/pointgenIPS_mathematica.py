"""IPS Mathematica point generator: cymetric-compatible Python wrapper for
the Improved Point Sampling algorithm implemented in
``PointGeneratorMathematicaCICYIPS.m``.

This module is the entry point for IPS-based CY point sampling. It opens a
``WolframLanguageSession``, loads the IPS sampler .m file, calls
``GeneratePointsMCICYIPS`` through wolframclient, and packages the results as

  * raw CSV/JSON artifacts mirroring the legacy Mathematica runner output
    (``points_real.csv``, ``points_imag.csv``, ``weights.csv``, ``omegas.csv``,
    ``kappas.csv``, ``patches_local.csv``, ``patches_global.csv``,
    ``j_elim_global.csv``, ``region_labels.csv``, ``acceptances.csv``,
    ``num_samples.csv``, ``L_matrices.json``, ``metadata.json``),
  * cymetric-standard training data (``dataset.npz``, ``basis.pickle``,
    ``point_gen.pickle``),
  * IPS-specific extras (``mathematica_extras.npz``, ``pack_summary.json``).

Drives Mathematica from Python, matching the cymetric pattern. Runner scripts
in ``run_IPS_scripts/`` instantiate this class with a CICY geometry and call
``prepare_dataset``.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from wolframclient.deserializers import binary_deserialize
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import Global as wlGlobal
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export as wlexport

from cymetric.pointgen.pointgen_mathematica import (
    ComplexFunctionConsumer,
    PointGeneratorMathematica,
)


IPS_M_FILE = str(Path(__file__).resolve().parent / "PointGeneratorMathematicaCICYIPS.m")

logger = logging.getLogger("pointgenIPS_mathematica")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(name)s:%(levelname)s:%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _resolve_kernel_path(explicit_path: Optional[str]) -> Optional[str]:
    """Find a Wolfram kernel binary.

    wolframclient's auto-detection only searches a hardcoded list of install
    locations; on HPC with ``module load mathematica`` the binary lives
    elsewhere on PATH. Resolve via ``shutil.which`` so module-loaded
    installations work without an explicit ``--kernel-path``.
    """
    if explicit_path:
        return explicit_path
    for name in ("WolframKernel", "wolfram", "MathKernel"):
        p = shutil.which(name)
        if p:
            return p
    return None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _patch_globals_from_local(patch_local: np.ndarray, ambient: np.ndarray) -> np.ndarray:
    """Mirror Mathematica's ``patchGlobalsFromLocal``.

    ``patch_local`` is an ``(npts, n_blocks)`` 1-indexed array of local patch
    indices; ``ambient`` is the projective dimensions vector. Returns
    ``(npts, n_blocks)`` of 1-indexed flattened global coordinate indices.
    """
    ambient = np.asarray(ambient, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(ambient + 1)[:-1]])
    return patch_local + offsets[None, :]


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr[:, None] if arr.ndim == 1 else arr


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class IPSPointGeneratorMathematica(PointGeneratorMathematica):
    """IPS point generator backed by ``GeneratePointsMCICYIPS`` in Mathematica.

    Inherits from cymetric's ``PointGeneratorMathematica`` so the standard
    cymetric machinery (``pullbacks``, ``fubini_study_metrics``,
    ``prepare_basis``) works unchanged. The ``generate_ips`` method drives the
    IPS sampler over a ``WolframLanguageSession`` and stores the full result
    set as ``ips_*`` attributes. ``prepare_dataset`` runs sampling and writes
    every artifact downstream code needs.
    """

    def __init__(
        self,
        monomials,
        coefficients,
        kmoduli,
        ambient,
        precision: int = 20,
        verbose: int = 1,
        vol_j_norm: Optional[float] = None,
        kernel_path: Optional[str] = None,
    ):
        super().__init__(
            monomials,
            coefficients,
            kmoduli,
            ambient,
            precision=precision,
            verbose=verbose,
            vol_j_norm=vol_j_norm,
            kernel_path=kernel_path,
        )

        # Populated by generate_ips
        self.ips_points: Optional[np.ndarray] = None
        self.ips_weights: Optional[np.ndarray] = None
        self.ips_omegas: Optional[np.ndarray] = None
        self.ips_patches_local: Optional[np.ndarray] = None   # 1-indexed
        self.ips_patches_global: Optional[np.ndarray] = None  # 1-indexed
        self.ips_j_elim_global: Optional[np.ndarray] = None   # 1-indexed
        self.ips_region_labels: Optional[np.ndarray] = None   # 1-indexed
        self.ips_kappas: Optional[np.ndarray] = None
        self.ips_acceptances: Optional[np.ndarray] = None
        self.ips_num_samples: Optional[np.ndarray] = None
        self.ips_dim_cy: Optional[int] = None
        self.ips_Ls: Optional[list] = None
        self.ips_total_pts_requested: Optional[int] = None
        self.ips_num_regions: Optional[int] = None
        self.ips_front_end: bool = False

    # ----- sampling --------------------------------------------------------

    def generate_ips(
        self,
        total_pts: int,
        num_regions: int,
        *,
        nproc: int = -1,
        front_end: bool = False,
    ) -> dict:
        """Call the IPS Mathematica sampler and parse its result.

        Populates ``self.ips_*`` attributes and returns a dict view of them.
        ``nproc=-1`` (default) uses ``SLURM_CPUS_PER_TASK`` if set, otherwise
        ``$ProcessorCount``.
        """
        if num_regions < 1:
            raise ValueError(f"num_regions must be >= 1; got {num_regions}")
        if total_pts < 1:
            raise ValueError(f"total_pts must be >= 1; got {total_pts}")

        self.ips_total_pts_requested = int(total_pts)
        self.ips_num_regions = int(num_regions)
        self.ips_front_end = bool(front_end)

        ambient_arg = self.ambient.tolist()
        coeffs_arg = [c.astype(np.complex128).tolist() for c in self.coefficients]
        monos_arg = [m.astype(np.int64).tolist() for m in self.monomials]
        kmoduli_arg = np.asarray(self.kmoduli, dtype=np.float64).tolist()

        resolved_kernel = _resolve_kernel_path(self.kernel_path)
        if resolved_kernel is None:
            raise RuntimeError(
                "Could not locate a Wolfram kernel binary. Either `module load "
                "mathematica/<version>` before launching, or pass an explicit "
                "--kernel-path."
            )
        logger.info(f"Opening WolframLanguageSession (kernel={resolved_kernel})")
        with WolframLanguageSession(kernel=resolved_kernel) as session:
            self.wl_session = session
            session.evaluate(wlexpr("ClientLibrary`SetErrorLogLevel[]"))

            logger.info(f"Loading IPS sampler: {IPS_M_FILE}")
            session.evaluate(wlexpr(f'Get["{IPS_M_FILE}"]'))

            self._launch_kernels(session, nproc)

            ips_fn = session.function(wlGlobal.GeneratePointsMCICYIPS)
            logger.info(
                f"Calling GeneratePointsMCICYIPS[total={total_pts}, regions={num_regions}, "
                f"precision={self.precision}, verbose={self.verbose}, frontEnd={front_end}]"
            )
            raw = ips_fn(
                int(total_pts),
                int(num_regions),
                ambient_arg,
                coeffs_arg,
                monos_arg,
                kmoduli_arg,
                int(self.precision),
                int(self.verbose),
                bool(front_end),
            )

            wxf = wlexport(raw, target_format="wxf")
            parsed = binary_deserialize(wxf, consumer=ComplexFunctionConsumer())

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 11:
            raise RuntimeError(
                f"Expected 11-tuple from GeneratePointsMCICYIPS; got {type(parsed).__name__} "
                f"of length {len(parsed) if hasattr(parsed, '__len__') else '?'}"
            )

        (
            points, weights, omegas, patches_local, j_elim_global,
            region_labels, kappas, acceptances, num_samples,
            dim_cy_box, Ls,
        ) = parsed[:11]

        self.ips_points = np.array(points, dtype=np.complex128)
        self.ips_weights = np.array(weights, dtype=np.float64).reshape(-1)
        self.ips_omegas = np.array(omegas, dtype=np.float64).reshape(-1)
        self.ips_patches_local = _ensure_2d(np.array(patches_local, dtype=np.int64))
        self.ips_j_elim_global = _ensure_2d(np.array(j_elim_global, dtype=np.int64))
        self.ips_region_labels = np.array(region_labels, dtype=np.int64).reshape(-1)
        self.ips_kappas = np.array(kappas, dtype=np.float64).reshape(-1)
        self.ips_acceptances = np.array(acceptances, dtype=np.float64).reshape(-1)
        self.ips_num_samples = np.array(num_samples, dtype=np.float64).reshape(-1)
        self.ips_dim_cy = int(np.asarray(dim_cy_box).flatten()[0])
        self.ips_Ls = [
            [np.array(blk, dtype=np.complex128) for blk in region_blocks]
            for region_blocks in Ls
        ]
        self.ips_patches_global = _patch_globals_from_local(
            self.ips_patches_local, self.ambient
        )

        logger.info(
            f"IPS sampling returned {len(self.ips_points)} points "
            f"across {len(self.ips_kappas)} region(s)."
        )
        return {
            "points": self.ips_points,
            "weights": self.ips_weights,
            "omegas": self.ips_omegas,
            "patches_local": self.ips_patches_local,
            "patches_global": self.ips_patches_global,
            "j_elim_global": self.ips_j_elim_global,
            "region_labels": self.ips_region_labels,
            "kappas": self.ips_kappas,
            "acceptances": self.ips_acceptances,
            "num_samples": self.ips_num_samples,
            "dim_cy": self.ips_dim_cy,
            "Ls": self.ips_Ls,
        }

    def _launch_kernels(self, session: WolframLanguageSession, nproc: int):
        if nproc is None or nproc < 0:
            slurm = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm:
                session.evaluate(wlexpr(f"LaunchKernels[{int(slurm)}]"))
                logger.info(f"Launched {int(slurm)} subkernels (SLURM_CPUS_PER_TASK).")
            else:
                session.evaluate(wlexpr("LaunchKernels[$ProcessorCount]"))
                n = session.evaluate(wlexpr("Length[Kernels[]]"))
                logger.info(f"Launched {n} subkernels ($ProcessorCount).")
        else:
            session.evaluate(wlexpr(f"LaunchKernels[{int(nproc)}]"))
            logger.info(f"Launched {int(nproc)} subkernels.")

    # ----- cymetric overrides ---------------------------------------------

    def generate_points(self, n_p, nproc=-1, **kwargs):
        """Return the first ``n_p`` IPS-sampled points."""
        if self.ips_points is None:
            raise RuntimeError(
                "No IPS data loaded. Call generate_ips(total_pts, num_regions) "
                "or prepare_dataset(...) first."
            )
        if n_p > len(self.ips_points):
            raise ValueError(
                f"Requested {n_p} points but only {len(self.ips_points)} sampled."
            )
        return self.ips_points[:n_p]

    def generate_point_weights(self, n_pw, omega=False, normalize_to_vol_j=False):
        """Return IPS-precomputed weights/omegas as a structured array.

        ``normalize_to_vol_j`` is accepted for interface compatibility but
        ignored: IPS weights already carry the region-aware kappa normalization
        from the Mathematica sampler.
        """
        if self.ips_points is None:
            raise RuntimeError("No IPS data loaded; call generate_ips() first.")
        n = min(n_pw, len(self.ips_points))
        fields = [("point", np.complex128, self.ncoords), ("weight", np.float64)]
        if omega:
            fields.append(("omega", np.complex128))
        out = np.zeros(n, dtype=np.dtype(fields))
        out["point"] = self.ips_points[:n]
        out["weight"] = self.ips_weights[:n]
        if omega:
            # Mathematica stores |Omega|^2 (real). Encode as sqrt(...)+0i so
            # downstream omega*conj(omega) recovers |Omega|^2 unchanged.
            out["omega"] = np.sqrt(
                np.maximum(self.ips_omegas[:n], 0.0)
            ).astype(np.complex128)
        return out

    # ----- packaging -------------------------------------------------------

    def prepare_dataset(
        self,
        output_dir: Union[str, Path],
        total_pts: int,
        num_regions: int,
        *,
        val_split: float = 0.1,
        shuffle: bool = False,
        seed: int = 2025,
        nproc: int = -1,
        front_end: bool = False,
    ) -> float:
        """End-to-end: sample with IPS and write every output file.

        Writes (all under ``output_dir``):

        * ``points_real.csv``, ``points_imag.csv``, ``weights.csv``, ``omegas.csv``,
          ``kappas.csv``, ``patches_local.csv``, ``patches_global.csv``,
          ``j_elim_global.csv``, ``region_labels.csv``, ``acceptances.csv``,
          ``num_samples.csv``, ``L_matrices.json``, ``metadata.json``
        * ``dataset.npz`` (X_train, y_train, X_val, y_val, val_pullbacks)
        * ``basis.pickle``, ``point_gen.pickle``, ``mathematica_extras.npz``,
          ``pack_summary.json``

        Returns the kappa value used for the basis.
        """
        if shuffle and num_regions > 1:
            raise ValueError(
                "shuffle=True is unsafe when num_regions > 1: it breaks "
                "region/point alignment relied on by region-weighted integration."
            )

        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)

        if self.ips_points is None:
            self.generate_ips(total_pts, num_regions, nproc=nproc, front_end=front_end)

        self._filter_invalid_rows()
        self._write_raw_outputs(d)
        kappa_for_basis = self._write_cymetric_outputs(
            d, val_split=val_split, shuffle=shuffle, seed=seed
        )
        return kappa_for_basis

    # ----- internals -------------------------------------------------------

    def _filter_invalid_rows(self):
        p = self.ips_points
        mask = (
            np.all(np.isfinite(p.real), axis=1)
            & np.all(np.isfinite(p.imag), axis=1)
            & np.isfinite(self.ips_weights)
            & np.isfinite(self.ips_omegas)
            & (self.ips_weights > 0)
            & (self.ips_omegas >= 0)
        )
        if np.all(mask):
            return
        n_bad = int(np.count_nonzero(~mask))
        logger.warning(f"Filtering {n_bad} invalid IPS rows before export.")
        self.ips_points = self.ips_points[mask]
        self.ips_weights = self.ips_weights[mask]
        self.ips_omegas = self.ips_omegas[mask]
        self.ips_patches_local = self.ips_patches_local[mask]
        self.ips_patches_global = self.ips_patches_global[mask]
        self.ips_j_elim_global = self.ips_j_elim_global[mask]
        self.ips_region_labels = self.ips_region_labels[mask]

    def _write_raw_outputs(self, d: Path):
        np.savetxt(d / "points_real.csv", self.ips_points.real, delimiter=",")
        np.savetxt(d / "points_imag.csv", self.ips_points.imag, delimiter=",")
        np.savetxt(d / "weights.csv", self.ips_weights, delimiter=",")
        np.savetxt(d / "omegas.csv", self.ips_omegas, delimiter=",")
        np.savetxt(d / "kappas.csv", self.ips_kappas, delimiter=",")
        np.savetxt(d / "patches_local.csv", self.ips_patches_local, delimiter=",", fmt="%d")
        np.savetxt(d / "patches_global.csv", self.ips_patches_global, delimiter=",", fmt="%d")
        np.savetxt(d / "j_elim_global.csv", self.ips_j_elim_global, delimiter=",", fmt="%d")
        np.savetxt(d / "region_labels.csv", self.ips_region_labels, delimiter=",", fmt="%d")
        np.savetxt(d / "acceptances.csv", self.ips_acceptances, delimiter=",")
        np.savetxt(d / "num_samples.csv", self.ips_num_samples, delimiter=",")

        l_matrices_assoc = []
        for r, region_blocks in enumerate(self.ips_Ls, start=1):
            l_matrices_assoc.append({
                "region": r,
                "blocks": [
                    {"real": L.real.tolist(), "imag": L.imag.tolist()}
                    for L in region_blocks
                ],
            })
        l_json = {
            "num_regions": len(self.ips_Ls),
            "num_blocks_per_region": (len(self.ips_Ls[0]) if self.ips_Ls else 0),
            "block_sizes": [int(a + 1) for a in self.ambient],
            "L_matrices": l_matrices_assoc,
        }
        with open(d / "L_matrices.json", "w") as f:
            json.dump(l_json, f, indent=2)

        metadata = self._build_metadata_dict()
        with open(d / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote raw IPS outputs to {d}")

    def _build_metadata_dict(self) -> dict:
        return {
            "schema_version": 3,
            "generator": "GeneratePointsMCICYIPS",
            "requested_total_points": int(self.ips_total_pts_requested),
            "num_regions_requested": int(self.ips_num_regions),
            "precision": int(self.precision),
            "verbose": int(self.verbose),
            "front_end": bool(self.ips_front_end),
            "num_points_valid": int(len(self.ips_points)),
            "point_dimension": int(self.ncoords),
            "dim_cy": int(self.ips_dim_cy),
            "num_hypersurfaces": int(len(self.coefficients)),
            "num_ambient_factors": int(len(self.ambient)),
            "dimPs": self.ambient.tolist(),
            "ambient": self.ambient.tolist(),
            "exponents": [m.tolist() for m in self.monomials],
            "coefficients_realimag": [
                [{"re": float(c.real), "im": float(c.imag)} for c in eq.astype(np.complex128)]
                for eq in self.coefficients
            ],
            "kahler_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "target_volume": (
                None if getattr(self, "vol_j_norm", None) is None
                else float(self.vol_j_norm)
            ),
            "omega_quantity": "|Omega|^2",
            "omega_description": (
                "Stored omegas are |Omega|^2 (real, nonnegative)."
            ),
            "weights_quantity": (
                "kappa * (|Omega|^2 / top_form_det) with IPS normalization"
            ),
            "patches_local_convention": (
                "1-indexed patch index within each projective block"
            ),
            "patches_global_convention": (
                "1-indexed flattened global coordinate indices"
            ),
            "j_elim_global_convention": (
                "1-indexed flattened global eliminated coordinate indices"
            ),
            "region_labels_convention": (
                "1-indexed metric/region label for each accepted point"
            ),
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
                "L_matrices_json": "L_matrices.json",
            },
        }

    def _write_cymetric_outputs(
        self, d: Path, val_split: float, shuffle: bool, seed: int
    ) -> float:
        n = len(self.ips_points)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        t_i = int((1.0 - val_split) * n)
        train_idx, val_idx = idx[:t_i], idx[t_i:]

        points = self.ips_points
        X = np.concatenate([points.real, points.imag], axis=1).astype(np.float64)
        y = np.column_stack([self.ips_weights, self.ips_omegas]).astype(np.float64)

        logger.info(f"Computing val_pullbacks for {len(val_idx)} validation points...")
        val_pullbacks = self.pullbacks(points[val_idx])

        dataset_path = d / "dataset.npz"
        np.savez_compressed(
            dataset_path,
            X_train=X[train_idx], y_train=y[train_idx],
            X_val=X[val_idx], y_val=y[val_idx],
            val_pullbacks=val_pullbacks,
        )
        logger.info(f"Wrote {dataset_path}")

        if len(self.ips_kappas) == 1:
            kappa_for_basis = float(self.ips_kappas[0])
        elif len(self.ips_kappas) > 1:
            kappa_for_basis = float(np.mean(self.ips_kappas))
            logger.info(
                f"Multiple kappas; using mean={kappa_for_basis:.6e} for basis."
            )
        else:
            kappa_for_basis = float(
                self.compute_kappa(points, self.ips_weights, self.ips_omegas)
            )
        self.prepare_basis(str(d), kappa=kappa_for_basis)
        logger.info(f"Wrote basis.pickle (kappa={kappa_for_basis:.6e})")

        # point_gen.pickle (cymetric pattern). Strip the wl session before pickling.
        old_session, self.wl_session = self.wl_session, None
        try:
            with open(d / "point_gen.pickle", "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Wrote {d/'point_gen.pickle'}")
        finally:
            self.wl_session = old_session

        order = np.concatenate([train_idx, val_idx])
        extras = {
            "patches_local_mathematica_1idx": self.ips_patches_local[order],
            "patches_local_python_0idx": self.ips_patches_local[order] - 1,
            "patches_global_mathematica_1idx": self.ips_patches_global[order],
            "patches_global_python_0idx": self.ips_patches_global[order] - 1,
            "j_elim_global_mathematica_1idx": self.ips_j_elim_global[order],
            "j_elim_global_python_0idx": self.ips_j_elim_global[order] - 1,
            "region_labels_mathematica_1idx": self.ips_region_labels[order],
            "region_labels_python_0idx": self.ips_region_labels[order] - 1,
            "acceptances": self.ips_acceptances,
            "num_samples": self.ips_num_samples,
        }
        if self.ips_Ls:
            n_regions_L = len(self.ips_Ls)
            n_blocks = len(self.ips_Ls[0])
            for r, region_Ls in enumerate(self.ips_Ls):
                for b, L in enumerate(region_Ls):
                    h = np.conj(L.T) @ L
                    extras[f"h_block_r{r}_b{b}_real"] = h.real.astype(np.float64)
                    extras[f"h_block_r{r}_b{b}_imag"] = h.imag.astype(np.float64)
            extras["h_blocks_num_regions"] = np.array([n_regions_L], dtype=np.int64)
            extras["h_blocks_num_blocks"] = np.array([n_blocks], dtype=np.int64)
        np.savez_compressed(d / "mathematica_extras.npz", **extras)
        logger.info(f"Wrote {d/'mathematica_extras.npz'}")

        summary = {
            "num_regions": int(self.ips_num_regions),
            "n_points_used": int(n),
            "n_coords": int(self.ncoords),
            "val_split": float(val_split),
            "shuffle": bool(shuffle),
            "seed": int(seed),
            "ambient": self.ambient.tolist(),
            "dimPs": self.ambient.tolist(),
            "vol_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "kahler_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "omega_quantity_from_metadata": "|Omega|^2",
            "kappa_for_basis": float(kappa_for_basis),
            "dataset_file": "dataset.npz",
            "basis_file": "basis.pickle",
            "has_region_labels": True,
            "has_acceptances": True,
            "has_num_samples": True,
            "has_h_blocks": bool(self.ips_Ls),
        }
        with open(d / "pack_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote {d/'pack_summary.json'}")

        return kappa_for_basis


# --------------------------------------------------------------------------
# CLI helper for runner scripts
# --------------------------------------------------------------------------

def add_common_cli_args(parser):
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-regions", type=int, required=True)
    parser.add_argument("--total-points", type=int, required=True)
    parser.add_argument("--precision", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--kernel-path", default=None,
        help="Wolfram kernel path; falls back to auto-detection if omitted.",
    )
    parser.add_argument(
        "--nproc", type=int, default=-1,
        help="Number of Wolfram subkernels. -1 = SLURM_CPUS_PER_TASK or $ProcessorCount.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(
        "pointgenIPS_mathematica.py is a library; run one of the runner "
        "scripts in run_IPS_scripts/ instead."
    )
