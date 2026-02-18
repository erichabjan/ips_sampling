import numpy as np
import os
import pandas as pd
import tensorflow as tf
from cymetric.models.helper import prepare_basis as prepare_tf_basis

from pathlib import Path
import sys
rep_path = os.getcwd().replace('/IPS/Euler_number', '')
sys.path.append(rep_path)
import python_functions as pf


### Import data
cicy_name = 'quintic' #'multi_eq' 'quintic' 'bicubic'
number_regions = 1
base_path = f'/home/habjan.e/CY_metric/data/ips_mathematica_output/{cicy_name}/'

points_real = pd.read_csv(base_path + f"points_real_{number_regions}.csv", header=None).values
points_imag = pd.read_csv(base_path + f"points_imag_{number_regions}.csv", header=None).values

weights = pd.read_csv(base_path + f"weights_{number_regions}.csv", header=None).values.flatten()
omegas = pd.read_csv(base_path + f"omegas_{number_regions}.csv", header=None).values.flatten()
kappas = pd.read_csv(base_path + f"kappas_{number_regions}.csv", header=None).values.flatten()


### cymetric data
dirname = f'/home/habjan.e/CY_metric/data/{cicy_name}'

BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
BASIS = prepare_tf_basis(BASIS)
new_basis = {}
for key in BASIS:
    new_basis[key] = tf.cast(BASIS[key], dtype=tf.complex64)
BASIS = new_basis

pts_s = np.concatenate([points_real, points_imag], axis=1).astype(np.float32)
wo_s = np.concatenate([weights[:, None], omegas[:, None]], axis=1).astype(np.float32)


### Monomials and coefficients
if cicy_name == 'multi_eq':

    monomials_eq1 = np.array([
        [3, 0, 0, 0,  0, 0, 0, 0],
        [2, 1, 0, 0,  0, 0, 0, 0],
        [0, 1, 1, 1,  0, 0, 0, 0],
        [0, 0, 3, 0,  0, 0, 0, 0],
    ], dtype=np.int64)

    monomials_eq2 = np.array([
        [0, 0, 0, 0,  3, 0, 0, 0],
        [0, 0, 0, 0,  2, 1, 0, 0],
        [0, 0, 0, 0,  0, 1, 1, 1],
        [0, 0, 0, 0,  0, 0, 0, 3],
    ], dtype=np.int64)

    monomials_eq3 = np.array([
        [1, 0, 0, 0,  1, 0, 0, 0],
        [0, 1, 0, 0,  0, 1, 0, 0],
        [0, 0, 1, 0,  0, 0, 1, 0],
        [0, 0, 0, 1,  0, 0, 0, 1],
    ], dtype=np.int64)

    monomials = [monomials_eq1, monomials_eq2, monomials_eq3]

if cicy_name == 'bicubic':

    monomials_eq1 = np.array([
        [3, 0, 0, 0, 3, 0],
        [0, 3, 0, 3, 0, 0],
        [2, 1, 0, 1, 2, 0],
        [1, 2, 0, 2, 1, 0],
        [1, 0, 2, 0, 1, 2],
        [0, 1, 2, 1, 0, 2],
    ], dtype=np.int64)

    monomials = [monomials_eq1]


if cicy_name == 'quintic':

    monomials = [np.array([
        [5,0,0,0,0],   
        [0,5,0,0,0],   
        [0,0,5,0,0], 
        [0,0,0,5,0], 
        [0,0,0,0,5],
        [1,1,1,1,1],
    ], dtype=np.int64)]

### Build the metric model
comp_model = pf.SpectralFSModelComp(None, BASIS = BASIS, alpha=[1.]*5, deg=2, monomials=monomials)


### Compute curvature (Riemann tensor)
riemann_path = Path(base_path) / f"riemann_pts_{cicy_name}_{number_regions}.pickle"
if riemann_path.is_file():
    riemann_path.unlink()
riemann_tensor = pf.compute_riemann(riemann_path, pts_s, comp_model, batch_size=512)


### Chern classes
c1, c2, c3, c3_form = pf.get_chern_classes(riemann_tensor, comp_model)


### Naive $\chi$ from your samples
chi_naive = tf.math.real(pf.integrate_native(c3_form, pts_s, wo_s, comp_model, normalize_to_vol=5.))


### $\chi$ for IPS wieghted by $\kappa$
regional_variances = pf.analyze_regions(c3_form, pts_s, wo_s, comp_model, kappas, verbose=False)
chi_weighted = tf.math.real(pf.integrate_variance_kappa_weighted(c3_form, pts_s, wo_s, comp_model, variances=regional_variances, kappas=kappas, normalize_to_vol=5.))


### Compare with true value
if cicy_name == 'multi_eq':
    chi_true = pf.euler_cicy([3, 3], [[3, 0], [0, 3], [1, 1]], check_cy=True)

if cicy_name == 'bicubic':
    chi_true = pf.euler_cicy([2, 2], [[3, 3]], check_cy=True)
    
if cicy_name == 'quintic':
    chi_true = pf.euler_cicy([4], [[5]], check_cy=True)


per_err = abs((chi_weighted.numpy() - chi_true) / chi_true) * 100

print(f'The true Euler number is: {chi_true}')
print(f'The Euler number from sampling is: {chi_weighted.numpy()}')
print(f'The percent error is: {per_err}')


### Save data as a numpy array
save_arr = np.array([chi_true, chi_weighted.numpy(), chi_naive.numpy(), per_err]).astype(np.float64)
np.save(base_path + f"Euler_Number_{number_regions}.npy", save_arr)