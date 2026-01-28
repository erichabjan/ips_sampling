import numpy as np
import mpmath as mp
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator


### Use Cymetric to generate data on the Weierstrass cubic in $\mathbb{P}^2$


fname = "/Users/erich/Downloads/Northeastern/ips_home/Data/weierstrass"

# Ambient space dimension
ambient = np.array([2], dtype=np.int64)

# Exponents for each monomial 
monomials = [np.array([
    [1,0,2],
    [0,3,0],
    [2,1,0],
], dtype=np.int64)]

# Coefficients for the monomials, in the same order:
# F = g_0 * (Z2^2 Z0) + g_1 * (Z1^3) + g_2 * (Z1 Z0^2)
a = 1.0
b = -4.0
c = 3.151212 * 60          # 60 * G4(i) \approx 60 * 3.151212
shape_moduli = [np.array([a, b, c], dtype=np.complex128)]

# Kähler/volume moduli (one for ℙ^2)
vol_moduli = np.array([1.0], dtype=np.float64)

if __name__ == "__main__":

    pg = CICYPointGenerator(monomials, shape_moduli, vol_moduli, ambient)
    kappa = pg.prepare_dataset(200_000, fname)
    pg.prepare_basis(fname, kappa=kappa)

    num_points = 10**4
    Z = pg.generate_points(num_points)


    ### Save the points to a numpy file in the file created above
    np.save(fname + f'/weierstrass_sampling.npy', Z)

    print('Weierstrass Cubic sampled and saved!')