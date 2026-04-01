import numpy as np
import os
import pandas as pd
import tensorflow as tf
from cymetric.models.helper import prepare_basis as prepare_tf_basis

import json
import argparse
from pathlib import Path
import sys
from typing import Optional, Union, List
rep_path = '/home/habjan.e/CY_metric/ips_sampling'
sys.path.append(rep_path)
import python_functions as pf

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator

def parse_args():
    p = argparse.ArgumentParser(description="Pack Mathematica IPS outputs into cymetric-compatible dataset.npz + basis.pickle")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing Mathematica CSV/JSON outputs")
    p.add_argument("--num-regions", type=int, required=True,
                   help="Region count used in Mathematica export filename suffixes")
    return p.parse_args()

# IPS outputs
args = parse_args()
num_regs = args.num_regions
dirname = str(args.input_dir)

### Import basis file

BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
BASIS = prepare_tf_basis(BASIS)

new_basis = {}
for key in BASIS:
    new_basis[key] = tf.cast(BASIS[key], dtype=tf.complex64)
BASIS = new_basis

### Import points

data = np.load(os.path.join(dirname, 'dataset.npz'))

new_data = {}
for key in data:
    new_data[key] = data[key]
data = new_data

### Import eliminated indices

extras = np.load(os.path.join(dirname, 'mathematica_extras.npz'))

source_region_labels = extras["region_labels_python_0idx"]
source_region_labels = np.asarray(source_region_labels, dtype=np.int64).reshape(-1)

j_elim = extras['j_elim_global_python_0idx']

j_elim = np.asarray(j_elim, dtype=np.int64)

if j_elim.ndim == 1:
    j_elim = j_elim[:, None]

### Import Kappas

kappas = pd.read_csv(os.path.join(dirname, f"kappas_{num_regs}.csv"), header=None).values.flatten()

### Format points for Euler calculation

pts = np.concatenate([data['X_train'], data['X_val']], axis=0)
wo = np.concatenate([data['y_train'], data['y_val']], axis=0)

pack_summary_path = os.path.join(dirname, "pack_summary.json")
with open(pack_summary_path, "r") as f:
    pack_summary = json.load(f)

is_shuffled = bool(pack_summary.get("shuffle", False)) if pack_summary is not None else False
if is_shuffled and len(kappas) > 1:
    raise ValueError(
        "Region-weighted Euler integration requires region ordering, but dataset was shuffled. "
        "Repack with shuffle=False (or save region IDs and group by region explicitly)."
    )

### Import meta data for CICY

with open(os.path.join(dirname, f'metadata_{num_regs}.json'), "r") as f:
    metadata = json.load(f)

monomials = [np.array(eq, dtype=np.int64) for eq in metadata["exponents"]]

coeffs_eq0 = metadata["coefficients_realimag"][0]
coefficients = np.array(
    [complex(c["re"], c["im"]) for c in coeffs_eq0],
    dtype=np.complex128
)

ambient = np.array(metadata["ambient"], dtype=np.int64)

n_fold = int(metadata["dim_cy"])

kmoduli = np.array(metadata["kahler_moduli"], dtype=np.float64)

num_regs = metadata['num_regions_requested']

### Build alpha

ncoords_ambient = int(np.sum(ambient + 1))
alpha = [1.0] * ncoords_ambient

### Moduli volume

mod_vol = pf.cy_volume_from_intersections(dirname, num_regions=num_regs)

### Load h_blocks from extras (L matrices per region)

h_blocks_per_region = {}
has_h_blocks = (
    "h_blocks_num_regions" in extras
    and "h_blocks_num_blocks" in extras
)

if has_h_blocks:
    n_regions_L = int(extras["h_blocks_num_regions"][0])
    n_blocks_L = int(extras["h_blocks_num_blocks"][0])
    for r in range(n_regions_L):
        blocks = []
        for b in range(n_blocks_L):
            h_re = extras[f"h_block_r{r}_b{b}_real"]
            h_im = extras[f"h_block_r{r}_b{b}_imag"]
            blocks.append(h_re + 1j * h_im)
        h_blocks_per_region[r] = blocks
    print(f"Loaded h_blocks for {n_regions_L} regions, {n_blocks_L} block(s) each")
else:
    # No L matrices available — region 0 uses identity (standard FS)
    print("No h_blocks found in extras; using standard FS metric for all regions")
    unique_regs = np.unique(source_region_labels)
    for r in unique_regs:
        h_blocks_per_region[int(r)] = None


print("Computing ownership labels from the final metric family...")
owner_labels, owner_min_scores, owner_score_matrix, owner_metric_weights = \
    pf.compute_region_ownership_labels(
        pts=pts,
        wo=wo,
        h_blocks_per_region=h_blocks_per_region,
        BASIS=BASIS,
        alpha=alpha,
        deg=2,
        monomials=monomials,
        kappas=kappas,
        j_elim=j_elim,
        batch_size=512,
        ownership_tol=1e-6,
        return_details=True
    )

print("Owner label counts:")
for r in range(len(kappas)):
    print(f"  region {r}: {(owner_labels == r).sum()}")

print("Estimating region volume fractions f_r...")
f_r, source_point_weights = pf.estimate_region_volume_fractions(
    wo=wo,
    source_region_labels=source_region_labels,
    owner_labels=owner_labels,
    kappas=kappas
)

print("Estimated region volume fractions:")
for r, fr in enumerate(f_r):
    print(f"  f_{r} = {fr}")
print(f"  sum f_r = {np.sum(f_r)}")

# Compute Euler Characteristic

### Build the base metric model (for Chern class contraction via lc tensor)

comp_model = pf.SpectralFSModelComp(None, BASIS=BASIS, alpha=alpha, deg=2, monomials=monomials)

### Compute curvature and metric per-region using consistent deformed metrics

print("Computing Riemann tensor and CY metric using owner labels...")
riemann_tensor, g_cy = pf.compute_riemann_regional(
    pts, owner_labels, h_blocks_per_region,
    BASIS, alpha, deg=2, monomials=monomials,
    j_elim=j_elim, batch_size=512, return_metric=True
)

### Chern classes and normalized Euler scalar density

c1, c2, c3, c3_form, chi_density = pf.get_chern_classes(
    riemann_tensor, comp_model, g_cy=g_cy
)

chi_density_owned, f_r_check, source_point_weights = pf.integrate_euler_multi_region_density(
    chi_density=chi_density.numpy(),
    wo=wo,
    source_region_labels=source_region_labels,
    owner_labels=owner_labels,
    kappas=kappas,
    normalize_to_vol=mod_vol
)

chi_density_owned = np.real(chi_density_owned)

chi_top_form = tf.math.real(
    pf.integrate_euler_top_form_regional(
        c3_form, pts, wo, comp_model,
        region_labels=source_region_labels,
        kappas=kappas,
        normalize_to_vol=mod_vol
    )
)

### Compare with true value

ambient_cfg, config = pf.config_matrix_from_metadata(metadata)
chi_true = pf.euler_cicy(ambient_cfg, config, check_cy=True)

per_err_top = abs((chi_top_form.numpy() - chi_true) / chi_true) * 100
per_err_owned = abs((chi_density_owned - chi_true) / chi_true) * 100

print(f'The true Euler number is: {chi_true}')
print(f'The Euler number from top-form IPS sampling is: {chi_top_form.numpy()}')
print(f'The percent error (top-form) is: {per_err_top}')
print(f'The Euler number from owned-region density estimator is: {chi_density_owned}')
print(f'The percent error (owned-density) is: {per_err_owned}')

save_arr = np.array([
    chi_true,
    chi_top_form.numpy(),
    chi_density_owned,
    per_err_top,
    per_err_owned
]).astype(np.float64)

np.save(os.path.join(dirname, f"Euler_Number_{num_regs}.npy"), save_arr)