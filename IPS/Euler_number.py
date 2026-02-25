import numpy as np
import os
import pandas as pd
import tensorflow as tf
from cymetric.models.helper import prepare_basis as prepare_tf_basis

import json
import argparse
from pathlib import Path
import sys
rep_path = '/home/habjan.e/CY_metric/ips_sampling'
sys.path.append(rep_path)
import python_functions as pf

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

coefficients = []
for eq_coeffs in metadata["coefficients_realimag"]:
    coeff_arr = np.array(
        [complex(c["re"], c["im"]) for c in eq_coeffs],
        dtype=np.complex128
    )
    coefficients.append(coeff_arr)

ambient = np.array(metadata["ambient"], dtype=np.int64)

n_fold = int(metadata["dim_cy"])

if "vol_moduli" in metadata:
    kmoduli = np.array(metadata["vol_moduli"], dtype=np.float64)
else:
    kmoduli = np.ones(len(ambient), dtype=np.float64)

### Build alpha

ncoords_ambient = int(np.sum(ambient + 1))
alpha = [1.0] * ncoords_ambient

### Moduli volume 

mod_vol = metadata.get("target_volume", None)

# Compute Euler Characteristic

### Build the metric model

comp_model = pf.SpectralFSModelComp(None, BASIS = BASIS, alpha=alpha, deg=2, monomials=monomials)

### Compute curvature (Riemann tensor)

riemann_path = Path(dirname) / f"riemann_mypts_{num_regs}.pickle"

if riemann_path.is_file():
    riemann_path.unlink()

riemann_tensor = pf.compute_riemann(riemann_path, pts, comp_model, batch_size=512)

### Chern classes

c1, c2, c3, c3_form = pf.get_chern_classes(riemann_tensor, comp_model)

### Naive $\chi$ from your samples

chi_naive = tf.math.real(pf.integrate_native(c3_form, pts, wo, comp_model, normalize_to_vol=mod_vol))

### $\chi$ for IPS wieghted by $\kappa$

regional_variances = pf.analyze_regions(c3_form, pts, wo, comp_model, kappas, verbose=False)

chi_weighted = tf.math.real(pf.integrate_variance_kappa_weighted(c3_form, pts, wo, comp_model, variances=regional_variances, kappas=kappas, normalize_to_vol=mod_vol))

### Compare with true value

ambient, config = pf.config_matrix_from_metadata(metadata)

chi_true = pf.euler_cicy(ambient, config, check_cy=True)

per_err = abs((chi_weighted.numpy() - chi_true) / chi_true) * 100

print(f'The true Euler number is: {chi_true}')
print(f'The Euler number from sampling is: {chi_weighted.numpy()}')
print(f'The percent error is: {per_err}')

### Save data as a numpy array

save_arr = np.array([chi_true, chi_weighted.numpy(), chi_naive.numpy(), per_err]).astype(np.float64)
np.save(os.path.join(dirname, f"Euler_Number_{num_regs}.npy"), save_arr)