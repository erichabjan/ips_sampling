import numpy as np
import mpmath as mp
from cymetric.pointgen.pointgen import PointGenerator

quintic_type = 'dwork'

fname = f"/home/habjan.e/CY_metric/data/{quintic_type}_quintic"

# Ambient space dimension
ambient = np.array([4], dtype=np.int64)

# Exponents for each monomial 
monomials = np.array([
    [5,0,0,0,0],   # Z0^5
    [0,5,0,0,0],   # Z1^5
    [0,0,5,0,0],   # Z2^5
    [0,0,0,5,0],   # Z3^5
    [0,0,0,0,5],   # Z4^5
    [1,1,1,1,1],   # Z0 Z1 Z2 Z3 Z4
], dtype=np.int64)

# Coefficients for the monomials, in the same order:

### pick a nonzero t for a different qunitic, i.e. something other than the fermat
if quintic_type == 'fermat':
    t = 0

elif quintic_type == 'dwork':
    t = 100

print('This sampling of the Quintic uses t = ', t)

# F = 1*Z0^5 + 1*Z1^5 + ... + 1*Z4^5 + (-5*t)*(Z0 Z1 Z2 Z3 Z4)
shape_moduli = np.array([
    1.0,  # for Z0^5
    1.0,  # for Z1^5
    1.0,  # for Z2^5
    1.0,  # for Z3^5
    1.0,  # for Z4^5
    -5.0 * (t + 0j),  # for Z0 Z1 Z2 Z3 Z4
], dtype=np.complex128)

# Kähler/volume moduli (one for ℙ^2)
vol_moduli = np.array([1.0], dtype=np.float64)


pg = PointGenerator(monomials, shape_moduli, vol_moduli, ambient)
kappa = pg.prepare_dataset(200_000, fname)
pg.prepare_basis(fname, kappa=kappa)

num_points = 10**4
Z = pg.generate_points(num_points)


### Save the points to a numpy file in the file created above
np.save(f'/home/habjan.e/CY_metric/ips_sampling/cymetric/fermat_quintic/{quintic_type}_quintic_sampling.npy', Z)

if quintic_type == 'fermat':
    quintic_t = 'Fermat'

elif quintic_type == 'dwork':
    quintic_t = 'Dwork'

print(quintic_t + ' Quintic sampled and saved!')