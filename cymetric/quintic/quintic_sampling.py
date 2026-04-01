import argparse
import json
import numpy as np
from pathlib import Path
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample quintic using cymetric with geometry from an IPS metadata file. "
                    "Outputs Mathematica-format CSVs/JSON so the result can be packed by "
                    "cymetric_compatibility.py and fed into Euler_number.py."
    )
    p.add_argument("--ips-metadata", type=Path, required=True,
                   help="Path to IPS metadata_{N}.json containing monomials and coefficients")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write Mathematica-format CSV/JSON outputs")
    p.add_argument("--num-points", type=int, default=10000,
                   help="Number of points to sample")
    p.add_argument("--num-regions", type=int, default=1,
                   help="Number of regions (always 1 for cymetric baseline)")
    return p.parse_args()


def _complex_to_json(z):
    """Convert a complex number to the {re, im} dict used by Mathematica metadata."""
    return {"re": float(z.real), "im": float(z.imag)}


def main():
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    N = args.num_regions

    # ---- Read geometry from IPS metadata ----
    with open(args.ips_metadata) as f:
        ips_metadata = json.load(f)

    monomials = [np.array(eq, dtype=np.int64) for eq in ips_metadata["exponents"]]

    coefficients = []
    for eq_coeffs in ips_metadata["coefficients_realimag"]:
        c = np.array([complex(x["re"], x["im"]) for x in eq_coeffs], dtype=np.complex128)
        coefficients.append(c)

    ambient = np.array(ips_metadata["ambient"], dtype=np.int64)
    kahler_moduli = np.array(
        ips_metadata.get("kahler_moduli", np.ones(len(ambient)).tolist()),
        dtype=np.float64,
    )
    vol_moduli = np.ones(len(ambient), dtype=np.float64)
    dim_cy = int(ips_metadata.get("dim_cy", int(np.sum(ambient) - len(monomials))))

    print(f"[cymetric_sample] Geometry: ambient={ambient.tolist()}, dim_cy={dim_cy}")
    print(f"[cymetric_sample] Monomials shapes: {[m.shape for m in monomials]}")
    print(f"[cymetric_sample] Num coefficients per equation: {[len(c) for c in coefficients]}")

    # ---- Create point generator and sample ----
    pg = CICYPointGenerator(monomials, coefficients, vol_moduli, ambient)

    n_request = int(args.num_points * 1.05)  # slight oversample to account for filtering
    pwo = pg.generate_point_weights(n_request, omega=True)

    points = pwo['point']
    weights = pwo['weight']
    omega_complex = pwo['omega']
    omegas = np.real(omega_complex * np.conj(omega_complex))  # |Omega|^2

    # Trim to requested count
    n = min(len(points), args.num_points)
    points = points[:n]
    weights = weights[:n]
    omegas = omegas[:n]

    print(f"[cymetric_sample] Generated {n} valid points")

    # ---- Compute kappa ----
    kappa = pg.compute_kappa(points, weights, omegas)
    print(f"[cymetric_sample] kappa = {kappa}")

    # ---- j_elim: which coordinate to eliminate (0-indexed from cymetric) ----
    j_elim_0idx = pg._find_max_dQ_coords(points)
    if j_elim_0idx.ndim == 1:
        j_elim_1idx = j_elim_0idx + 1  # Mathematica 1-indexed
    else:
        j_elim_1idx = j_elim_0idx + 1

    # ---- Patches: index of max |coord| per projective factor ----
    coord_starts = np.concatenate([[0], np.cumsum(ambient + 1)])
    n_factors = len(ambient)

    if n_factors == 1:
        # Single projective factor: 1D arrays
        factor_coords = np.abs(points[:, coord_starts[0]:coord_starts[1]])
        local_idx = np.argmax(factor_coords, axis=1)
        patches_local = local_idx + 1               # 1-indexed within factor
        patches_global = coord_starts[0] + local_idx + 1  # 1-indexed global
    else:
        # Multiple factors: 2D arrays (one column per factor)
        patches_local = np.zeros((n, n_factors), dtype=np.int64)
        patches_global = np.zeros((n, n_factors), dtype=np.int64)
        for k in range(n_factors):
            start = coord_starts[k]
            end = coord_starts[k + 1]
            factor_coords = np.abs(points[:, start:end])
            local_idx = np.argmax(factor_coords, axis=1)
            patches_local[:, k] = local_idx + 1
            patches_global[:, k] = start + local_idx + 1

    # ---- Region labels: all 1 for single region (Mathematica 1-indexed) ----
    region_labels = np.ones(n, dtype=np.int64)

    # ---- Save CSVs ----
    points_real = points.real
    points_imag = points.imag

    np.savetxt(output_dir / f"points_real_{N}.csv", points_real, delimiter=",")
    np.savetxt(output_dir / f"points_imag_{N}.csv", points_imag, delimiter=",")
    np.savetxt(output_dir / f"weights_{N}.csv", weights, delimiter=",")
    np.savetxt(output_dir / f"omegas_{N}.csv", omegas, delimiter=",")
    np.savetxt(output_dir / f"kappas_{N}.csv", [kappa], delimiter=",")
    np.savetxt(output_dir / f"patches_local_{N}.csv", patches_local, delimiter=",", fmt="%d")
    np.savetxt(output_dir / f"patches_global_{N}.csv", patches_global, delimiter=",", fmt="%d")
    np.savetxt(output_dir / f"j_elim_global_{N}.csv", j_elim_1idx, delimiter=",", fmt="%d")
    np.savetxt(output_dir / f"region_labels_{N}.csv", region_labels, delimiter=",", fmt="%d")
    np.savetxt(output_dir / f"acceptances_{N}.csv", [n], delimiter=",")
    np.savetxt(output_dir / f"num_samples_{N}.csv", [n_request], delimiter=",")

    # ---- Save metadata JSON ----
    ncoords = int(np.sum(ambient + 1))
    metadata = {
        "schema_version": 2,
        "generator": "cymetric_CICYPointGenerator",
        "requested_total_points": args.num_points,
        "num_regions_requested": N,
        "precision": 16,
        "verbose": 1,
        "front_end": False,
        "num_points_valid": n,
        "point_dimension": ncoords,
        "dim_cy": dim_cy,
        "num_hypersurfaces": len(monomials),
        "num_ambient_factors": n_factors,
        "dimPs": ambient.tolist(),
        "ambient": ambient.tolist(),
        "exponents": [m.tolist() for m in monomials],
        "coefficients_realimag": [
            [_complex_to_json(c) for c in eq_c]
            for eq_c in coefficients
        ],
        "kahler_moduli": kahler_moduli.tolist(),
        "target_volume": None,
        "omega_quantity": "|Omega|^2",
        "omega_description": "cymetric omegas are |Omega|^2 = real(Omega * conj(Omega))",
        "weights_quantity": "|Omega|^2 / det(g_FS_norm) from cymetric point_weight",
        "patches_local_convention": "1-indexed patch index within each projective block",
        "patches_global_convention": "1-indexed flattened global coordinate indices",
        "j_elim_global_convention": "1-indexed flattened global eliminated coordinate indices",
        "region_labels_convention": "1-indexed metric/region label for each point (all 1 for cymetric)",
        "files": {
            "points_real_csv": f"points_real_{N}.csv",
            "points_imag_csv": f"points_imag_{N}.csv",
            "weights_csv": f"weights_{N}.csv",
            "omegas_csv": f"omegas_{N}.csv",
            "kappas_csv": f"kappas_{N}.csv",
            "patches_local_csv": f"patches_local_{N}.csv",
            "patches_global_csv": f"patches_global_{N}.csv",
            "j_elim_global_csv": f"j_elim_global_{N}.csv",
            "region_labels_csv": f"region_labels_{N}.csv",
            "acceptances_csv": f"acceptances_{N}.csv",
            "num_samples_csv": f"num_samples_{N}.csv",
        },
    }

    with open(output_dir / f"metadata_{N}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[cymetric_sample] Saved all outputs to {output_dir}")
    print(f"[cymetric_sample] Done. {n} points with {ncoords} coordinates each.")


if __name__ == "__main__":
    main()
