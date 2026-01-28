import numpy as np
import mpmath as mp
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator

fname = f"/Users/erich/Downloads/Northeastern/ips_home/Data/bicubic"

# Ambient space dimension
ambient = np.array([2, 2], dtype=np.int64)

# Exponents for each monomial 
monomials_eq1 = np.array([
    [3, 0, 0, 0, 3, 0],
    [0, 3, 0, 3, 0, 0],
    [2, 1, 0, 1, 2, 0],
    [1, 2, 0, 2, 1, 0],
    [1, 0, 2, 0, 1, 2],
    [0, 1, 2, 1, 0, 2],
], dtype=np.int64)

monomials = [monomials_eq1]

# Coefficients
coeff_eq1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.complex128)

coefficients = [coeff_eq1]

# Kahler moduli (one for each projective space)
vol_moduli = np.array([1.0, 1.0], dtype=np.float64)

if __name__ == "__main__":

    pg = CICYPointGenerator(monomials, coefficients, vol_moduli, ambient)
    kappa = pg.prepare_dataset(200_000, fname)
    pg.prepare_basis(fname, kappa=kappa)

    num_points = 10**4
    Z = pg.generate_points(num_points)


    ### Save the points to a numpy file in the file created above
    np.save(fname + f'/bicubic_sampling.npy', Z)

    print('Bicubic sampled and saved!')