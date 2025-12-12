import numpy as np
import mpmath as mp
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator

fname = f"/Users/erich/Downloads/Northeastern/ips_home/Data/multi_equation"

# Ambient space dimension
ambient = np.array([3, 3], dtype=np.int64)

# Exponents for each monomial 
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

# Coefficients
coeff_eq1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
coeff_eq2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
coeff_eq3 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)

coefficients = [coeff_eq1, coeff_eq2, coeff_eq3]

# Kahler moduli (one for each projective space)
vol_moduli = np.array([1.0, 1.0], dtype=np.float64)

if __name__ == "__main__":

    pg = CICYPointGenerator(monomials, coefficients, vol_moduli, ambient)
    kappa = pg.prepare_dataset(200_000, fname)
    pg.prepare_basis(fname, kappa=kappa)

    num_points = 10**4
    Z = pg.generate_points(num_points)


    ### Save the points to a numpy file in the file created above
    np.save(fname + f'/multi_equation_sampling.npy', Z)

    print('Multi-Equation CICY sampled and saved!')