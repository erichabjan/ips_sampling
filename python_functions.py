import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

import pickle
import math
import os
import pandas as pd
import seaborn as sns
from sympy import LeviCivita
from tqdm import tqdm
import tensorflow as tf
import sympy as sp

from cymetric.models.models import PhiFSModel
from cymetric.models.helper import prepare_basis as prepare_tf_basis

def weierstrass_uniform(Z, weierstrass_coeff):
    
    a, b, c = weierstrass_coeff[0], weierstrass_coeff[1], weierstrass_coeff[2]
    
    ### homogenize points 
    mask = np.abs(Z[:,0]) > 1e-14
    x = (Z[mask,1] / Z[mask,0]).astype(np.complex128)
    y = (Z[mask,2] / Z[mask,0]).astype(np.complex128)
    
    # --- constants for g3=0 (lemniscatic) ---
    c_real = float(np.real(c))       # c > 0 in your setup
    rt = mp.sqrt(c_real)             # √c (real, positive)
    a = rt/2                         # e1=+a, e2=0, e3=-a
    alpha = c_real**0.25             # α = (e1-e3)^{1/2} = c^{1/4}
    m = mp.mpf('0.5')                # k^2 = 1/2
    K  = mp.ellipk(m)
    Kp = mp.ellipk(1-m)
    
    # Jacobi helpers for the older mpmath API: ellipfun(kind, u, m)
    def jacobi_sn(u, m): return mp.ellipfun('sn', u, m)
    def jacobi_cn(u, m): return mp.ellipfun('cn', u, m)
    def jacobi_dn(u, m): return mp.ellipfun('dn', u, m)

    def inv_u(xi, yi, *, m=m, alpha=alpha, rt=rt, a=a):
        """Invert x -> u on y^2 = 4x^3 - c x (g3=0) using Jacobi sn."""
        # sn^2(αu) = √c / (x + √c/2)
        s2   = rt / (xi + a)          # can be complex
        s    = mp.sqrt(s2)            # principal branch
        Uhat = mp.ellipf(mp.asin(s), m)  # Uhat = α u  (inverse sn)

        # Predict y from Uhat to choose the correct sheet/sign
        sn = jacobi_sn(Uhat, m)
        cn = jacobi_cn(Uhat, m)
        dn = jacobi_dn(Uhat, m)
        y_pred = -(alpha**3) * (cn * dn) / (sn**3)

        u = Uhat / alpha
        if abs(y_pred - yi) > abs(-y_pred - yi):
            u = -u

        return complex(u)

    U = np.array([inv_u(xi, yi) for xi, yi in zip(x, y)], dtype=np.complex128)
    
    w1, w2 = K/alpha, 1j*Kp/alpha   # half-periods

    M = np.array([[ (2*w1).real, (2*w2).real ],
                  [ (2*w1).imag, (2*w2).imag ]], dtype=float)

    ab = (np.linalg.inv(M) @ np.vstack([U.real, U.imag])).T
    
    return ab[:,0], ab[:,1]




def sampling_plot(Z_in, q=0.995, eps=1e-14, s=1.0, alpha=0.8, color='k', equal_aspect=False):
    """
    Plot Re(Z_1/Z_0) vs Re(Z_k/Z_0) for k=2..M-1 as M-2 panels in a row.

    Parameters
    ----------
    Z_in : array-like, shape (N, M)
        Complex (or real) array with columns Z_0, Z_1, ..., Z_{M-1}.
    q : float, optional
        Central quantile to retain (e.g., 0.995 keeps the middle 99.5%).
        Used to clip both X (per panel) and Y (shared across panels).
    eps : float, optional
        Threshold to mask |Z_0| to avoid division by (near) zero.
    s, alpha, color : matplotlib scatter kwargs
    equal_aspect : bool
        If True, set each panel to equal aspect ('box').

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    Z = np.asarray(Z_in)
    if Z.ndim != 2:
        raise ValueError("Z_in must be a 2D array of shape (N, M).")
    N, M = Z.shape
    if M < 3:
        raise ValueError("Z_in must have at least 3 columns (Z_0, Z_1, Z_2).")

    # Mask to avoid dividing by tiny Z_0
    mask = np.abs(Z[:, 0]) > eps
    if not np.any(mask):
        raise ValueError("All rows have |Z_0| <= eps; nothing to plot.")
    Z = Z[mask]

    denom = Z[:, 0]
    Y = np.real(Z[:, 1] / denom)

    # Shared Y clipping across all panels
    lo_y, hi_y = np.quantile(Y, 1 - q), np.quantile(Y, q)

    ncols = M - 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, k in enumerate(range(2, M)):
        Xk = np.real(Z[:, k] / denom)
        # Per-panel X clipping
        lo_x, hi_x = np.quantile(Xk, 1 - q), np.quantile(Xk, q)

        axes[i].scatter(np.clip(Xk, lo_x, hi_x), np.clip(Y, lo_y, hi_y),
                        c=color, s=s, alpha=alpha)
        if equal_aspect:
            axes[i].set_aspect('equal', 'box')

        axes[i].set_xlabel(rf'Re$(Z_{k}/Z_0)$', fontsize=12)
        #axes[i].set_xlim(lo_x, hi_x)
        #axes[i].set_ylim(lo_y, hi_y)
        if i == 0:
            axes[i].set_ylabel(r'Re$(Z_1/Z_0)$', fontsize=12)

    plt.tight_layout()
    return fig, axes


def multi_sampling_plot(Z_in, q=0.995, eps=1e-14, s=1.0, alpha=0.8,
                  color='k', equal_aspect=False, n_segments=1, fontsize = 12):
    """
    Plot Re(Z_1/Z_0) vs Re(Z_k/Z_0) for k=2..M-1 as M-2 panels in a row.

    Parameters
    ----------
    Z_in : array-like, shape (N, M)
        Complex (or real) array with columns Z_0, Z_1, ..., Z_{M-1}.
    q : float, optional
        Central quantile to retain (e.g., 0.995 keeps the middle 99.5%).
        Used to clip both X (per panel) and Y (shared across panels).
    eps : float, optional
        Threshold to mask |Z_0| to avoid division by (near) zero.
    s, alpha, color : matplotlib scatter kwargs
        If color is None, Matplotlib's color cycle is used, which is
        useful when plotting multiple segments in different colors.
    equal_aspect : bool
        If True, set each panel to equal aspect ('box').
    n_segments : int
        Number of chunks to split the data into along the sample axis.
        Each chunk is plotted with a separate scatter call (different color).

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    Z = np.asarray(Z_in)
    if Z.ndim != 2:
        raise ValueError("Z_in must be a 2D array of shape (N, M).")
    N, M = Z.shape
    if M < 3:
        raise ValueError("Z_in must have at least 3 columns (Z_0, Z_1, Z_2).")

    # Mask to avoid dividing by tiny Z_0
    mask = np.abs(Z[:, 0]) > eps
    if not np.any(mask):
        raise ValueError("All rows have |Z_0| <= eps; nothing to plot.")
    Z = Z[mask]

    denom = Z[:, 0]
    Y = np.real(Z[:, 1] / denom)

    # Shared Y clipping across all panels
    lo_y, hi_y = np.quantile(Y, 1 - q), np.quantile(Y, q)

    # Prepare segmentation indices
    N_eff = Y.shape[0]
    if n_segments is None or n_segments < 1:
        n_segments = 1
    seg_size = int(np.ceil(N_eff / n_segments))
    segments = [slice(j, min(j + seg_size, N_eff))
                for j in range(0, N_eff, seg_size)]

    ncols = M - 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, k in enumerate(range(2, M)):
        Xk = np.real(Z[:, k] / denom)
        # Per-panel X clipping
        lo_x, hi_x = np.quantile(Xk, 1 - q), np.quantile(Xk, q)

        # Plot each segment separately to get multiple colors
        for seg in segments:
            X_seg = np.clip(Xk[seg], lo_x, hi_x)
            Y_seg = np.clip(Y[seg], lo_y, hi_y)
            if color is None:
                # Let Matplotlib use its color cycle
                axes[i].scatter(X_seg, Y_seg, s=s, alpha=alpha)
            else:
                axes[i].scatter(X_seg, Y_seg, c=color, s=s, alpha=alpha)

        if equal_aspect:
            axes[i].set_aspect('equal', 'box')

        axes[i].set_xlabel(rf'Re$(Z_{k}/Z_0)$', fontsize=fontsize)
        if i == 0:
            axes[i].set_ylabel(r'Re$(Z_1/Z_0)$', fontsize=fontsize)

    plt.tight_layout()
    return fig, axes


class SpectralFSModel(PhiFSModel):
    def __init__(self, *args, **kwargs):
        self.k = [kwargs['deg'] for _ in range(len(kwargs['BASIS']['AMBIENT']))]
        self.monomials = kwargs['monomials']
        del kwargs['deg']
        del kwargs['monomials']
        super(SpectralFSModel, self).__init__(*args, **kwargs)
        
        self._generate_sections(self.k)
        self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        self.learn_transition = tf.cast(False, dtype=tf.bool)
        self.learn_ricci = tf.cast(False, dtype=tf.bool)
        self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        self.learn_volk = tf.cast(False, dtype=tf.bool)

    def feature_engineered_call(self, pts):
        # assumes real points
        c_pts = tf.complex(pts[:, :pts.shape[-1]//2], pts[:, pts.shape[-1]//2:])
        eig_funcs = self.get_eigenfunction_basis(c_pts)
        eig_funcs = tf.concat((tf.math.real(eig_funcs), tf.math.imag(eig_funcs)), axis=-1) 
        return self.model(eig_funcs, training=False)

    
    def call(self, input_tensor, training=True, j_elim=None):
        # return self.fubini_study_pb(input_tensor, j_elim=j_elim)
        # nn prediction
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(input_tensor)
                # Need to disable training here, because batch norm and dropout mix the batches, such that batch_jacobian is no longer reliable.
                phi = self.feature_engineered_call(input_tensor)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor, experimental_use_pfor=False)
        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
            0.25*dd_phi[:, :self.ncoords, :self.ncoords], \
            0.25*dd_phi[:, :self.ncoords, self.ncoords:], \
            0.25*dd_phi[:, self.ncoords:, :self.ncoords], \
            0.25*dd_phi[:, self.ncoords:, self.ncoords:]
        dd_phi = tf.complex128(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum('xai,xij,xbj->xab', pbs, dd_phi, tf.math.conj(pbs))

        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return tf.math.add(fs_cont, dd_phi)
        return tf.transpose(tf.math.add(fs_cont, dd_phi), perm=[0,2,1])  # transpose to get g_{a\bar b} instead of g_{\bar a, b}

    @staticmethod
    def get_num_sections(n, k):
        return math.comb(n + k - 1, k) if k < n else math.comb(n + k - 1, k) - math.comb(k - 1, k - n)
    
    @staticmethod
    def get_levicivita_tensor(dim):
        lc = np.zeros(tuple([dim for _ in range(dim)]))
        for t in it.permutations(range(dim), r=dim):
            lc[t] = LeviCivita(*t)
        return tf.cast(lc, dtype=tf.complex128)
    
    def get_eigenfunction_basis(self, c_pts):
        # expects complex points
        c_pts = tf.cast(c_pts, tf.complex128)
        s_i = self.eval_sections_vec(c_pts)
        bs_j = self.eval_sections_vec(tf.math.conj(c_pts))
        sbs = tf.reshape(tf.einsum('xi,xj->xij', s_i, bs_j), (-1, s_i.shape[-1]**2))
        return tf.einsum('xa, x->xa', sbs, 1. / tf.einsum('xi,xi->x', c_pts, tf.math.conj(c_pts)) ** self.k)
    
    def generate_monomials(self, n, deg):
        if n == 1:
            yield (deg,)
        else:
            for i in range(deg + 1):
                for j in self.generate_monomials(n - 1, deg - i):
                    yield (i,) + j
    
    def _generate_sections(self, k, ambient=False):
        self.sections = None
        ambient_polys = [0 for i in range(len(k))]
        for i in range(len(k)):
            # create all monomials of degree k in ambient space factors
            ambient_polys[i] = list(self.generate_monomials(self.degrees[i], k[i]))
        # create all combinations for product of projective spaces
        monomial_basis = [x for x in ambient_polys[0]]
        for i in range(1, len(k)):
            lenB = len(monomial_basis)
            monomial_basis = monomial_basis*len(ambient_polys[i])
            for l in range(len(ambient_polys[i])):
                for j in range(lenB):
                    monomial_basis[l*lenB+j] = monomial_basis[l * lenB + j] + ambient_polys[i][l]
        sections = np.array(monomial_basis, dtype=np.int64)
        # reduce sections; pick (arbitrary) first monomial in point gen
        if not ambient:
            ### Handling if there is multiple defining equatiions
            if isinstance(self.monomials, list):
                first_mono = self.monomials[0][0]
            else:
                first_mono = self.monomials[0]
            reduced = np.unique(np.where(sections - first_mono < -0.1)[0])
            sections = sections[reduced]
        self.sections = tf.cast(sections, tf.complex128)
        self.nsections = len(self.sections)

    def eval_sections_vec(self, points):
        return tf.reduce_prod(tf.math.pow(tf.expand_dims(points, 1), self.sections), axis=-1)

class SpectralFSModelComp(SpectralFSModel):
    def __init__(self, *args, **kwargs):
        super(SpectralFSModelComp, self).__init__(*args, **kwargs)
        self.lc = tf.cast(self.lc, dtype=tf.complex64)
    
    def call(self, input_tensor, training=True, j_elim=None):
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

def christoffel(g_model, pts):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = g_model(pts)
        g_re, g_im = tf.math.real(g), tf.math.imag(g)
    d_g_re = tape.batch_jacobian(g_re, pts, experimental_use_pfor=False)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = g_model(pts)
        g_im = tf.math.imag(g)
    d_g_im = tape.batch_jacobian(g_im, pts, experimental_use_pfor=False)
    dx_g_re, dy_g_re, dx_g_im, dy_g_im = d_g_re[:,:,:,:g_model.ncoords], d_g_re[:,:,:,g_model.ncoords:], d_g_im[:,:,:,:g_model.ncoords], d_g_im[:,:,:,g_model.ncoords:]
    d_g = tf.complex(dx_g_re + dy_g_im, -dy_g_re + dx_g_im)
    pbs = g_model.pullbacks(pts)
    d_g_pb = tf.einsum('xbn,xcdn->xbcd', pbs, d_g)
    gamma = tf.einsum('xbcd, xda->xabc', d_g_pb, tf.linalg.inv(g))
    return gamma

def riemann(g_model, pts):  # return R^a_{b\bar c d}
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts)
        gamma_re = tf.math.real(gamma)
    d_gamma_re = tape.batch_jacobian(gamma_re, pts, experimental_use_pfor=False)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts)
        gamma_im =tf.math.imag(gamma)
    d_gamma_im = tape.batch_jacobian(gamma_im, pts, experimental_use_pfor=False)
    dx_gamma_re, dy_gamma_re, dx_gamma_im, dy_gamma_im = d_gamma_re[:,:,:,:,:g_model.ncoords], d_gamma_re[:,:,:,:,g_model.ncoords:], d_gamma_im[:,:,:,:,:g_model.ncoords], d_gamma_im[:,:,:,:,g_model.ncoords:]
    dbarc_gamma = tf.complex(dx_gamma_re - dy_gamma_im, dy_gamma_re + dx_gamma_im)
    pbs = g_model.pullbacks(pts)
    riem = -tf.einsum('xci,xabdi->xabcd', tf.math.conj(pbs), dbarc_gamma)
    return riem

# compute Ric from \del \bar\del \log \det g
def get_Ricci(g_model, pts):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(pts)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(pts)
            # Need to disable training here, because batch norm and dropout mix the batches, such that batch_jacobian is no longer reliable.
            ldg = tf.math.log(tf.linalg.det(g_model(pts)))
        d_ldg = tape2.gradient(ldg, pts)
    dd_ldg = tape1.batch_jacobian(d_ldg, pts, experimental_use_pfor=False)
    dx_dx_ldg, dx_dy_ldg, dy_dx_ldg, dy_dy_ldg = \
        0.25*dd_ldg[:, :g_model.ncoords, :g_model.ncoords], \
        0.25*dd_ldg[:, :g_model.ncoords, g_model.ncoords:], \
        0.25*dd_ldg[:, g_model.ncoords:, :g_model.ncoords], \
        0.25*dd_ldg[:, g_model.ncoords:, g_model.ncoords:]
    dd_ldg = tf.complex(dx_dx_ldg + dy_dy_ldg, dx_dy_ldg - dy_dx_ldg)
    pbs = g_model.pullbacks(pts)
    dd_ldg = tf.einsum('xai,xij,xbj->xab', pbs, dd_ldg, tf.math.conj(pbs))
    return dd_ldg


## Definition for kappa- and variance-weighted integrations
def compute_riemann(target_file, pts, comp_model, batch_size=10000):
    pts = tf.cast(pts, dtype=tf.float32)
    num_batches = math.ceil(len(pts) / batch_size)
    if not os.path.exists(target_file):
        for i in tqdm(range(num_batches)):
            if i == 0:
                riem = riemann(comp_model, pts[:batch_size])
            else:
                riem = tf.concat([riem, riemann(comp_model, pts[i * batch_size:(i + 1) * batch_size])], axis=0)
        with open(target_file, 'wb') as hnd:
            pickle.dump(riem.numpy(), hnd)
    else:
        with open(target_file, 'rb') as hnd:
            riem = pickle.load(hnd)
    riem = tf.cast(riem, dtype=tf.complex64)
    return riem

def get_chern_classes(riem, comp_model):
    # first Chern class
    tr_R = -tf.einsum('xaabc->xcb', riem)
    c1 = 1.j/(2 * math.pi) * tr_R
    
    # second Chern class
    tr_R2 = tf.einsum('xabmn, xbaop->xnmpo', riem, riem)
    c2 =  1./(2*(2*math.pi)**2) * (tr_R2 - tf.einsum('xab,xcd->xabcd', tr_R, tr_R))
    
    # thrid Chern class
    tr_R3 =  tf.einsum('xabmn, xbcop, xcaqr->xnmporq', riem, riem, riem)
    c3 = 1./3. * tf.einsum('xmn,xopqr->xmnopqr', c1, c2) + 1./(3*(2*math.pi)**2) * tf.einsum('xmn,xopqr->xmnopqr', c1, tr_R2) -1.j/(3*(2*math.pi)**3) * tr_R3
    
    c3_form = 1./math.factorial(comp_model.nfold) * tf.einsum('xmnopqr,moq,npr->x', c3, comp_model.lc, comp_model.lc)

    return c1, c2, c3, c3_form
    
def integrate_native(integrand, pts, wo, comp_model, normalize_to_vol=None):
    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    aux_weights = tf.repeat(tf.expand_dims(aux_weights, axis=0), repeats=[len(comp_model.BASIS['KMODULI'])], axis=0)

    res = (-1.j/2)**comp_model.nfold * tf.reduce_mean(integrand * aux_weights, axis=-1)[0]
    if normalize_to_vol is not None:
        vol = tf.abs(tf.reduce_mean(tf.linalg.det(comp_model(pts)) * aux_weights[0], axis=-1))
        kappa = normalize_to_vol / vol
        vol *= tf.cast(kappa, dtype=vol.dtype)
        res *= tf.cast(kappa, dtype=res.dtype)
    return res
    
def analyze_regions(integrand, pts, wo, comp_model, kappas, verbose=True):
    # Diagnostic tests to understand why importance sampling is worse

    # Test 1: Check if kappa values are reasonable
    if verbose:
        print("=== Kappa Analysis ===")
        print(f"Kappa values: {kappas}")
        print(f"Kappa std/mean: {np.std(kappas)/np.mean(kappas):.3f}")
    
    # Test 2: Check regional contribution variance
    if verbose:
        print("\n=== Regional Contribution Analysis ===")
    regional_contributions = []
    regional_variances = []
    
    num_pts_per_region = len(pts) // len(kappas)
    for i in range(len(kappas)):
        start_idx = num_pts_per_region * i
        end_idx = num_pts_per_region * (i + 1)
        
        # Get integrand values for this region
        region_integrand = integrand[start_idx:end_idx]
        region_weights = wo[start_idx:end_idx, 0] / wo[start_idx:end_idx, 1]
        
        # Raw contribution (without kappa weighting)
        raw_contribution = tf.reduce_mean(region_integrand * tf.cast(region_weights, dtype=tf.complex64))
        
        # Variance of integrand in this region
        integrand_var = tf.reduce_mean(tf.abs(region_integrand - tf.reduce_mean(region_integrand))**2)
        
        regional_contributions.append(raw_contribution.numpy())
        regional_variances.append(integrand_var.numpy())
        
        if verbose:
            print(f"Region {i}: contribution = {raw_contribution:.3f}, variance = {integrand_var:.6f}")
    
    if verbose:
        print(f"Regional contribution std: {np.std(regional_contributions):.3f}")
        print(f"If regions have very different contributions, IPS might help")
        print(f"If regions are similar, IPS adds noise without benefit")
    
    # Test 3: Check if importance weights are well-behaved
    if verbose:
        print("\n=== Importance Weight Analysis ===")
    for i in range(len(kappas)):
        start_idx = num_pts_per_region * i
        end_idx = num_pts_per_region * (i + 1)
        
        base_weights = wo[start_idx:end_idx, 0] / wo[start_idx:end_idx, 1]
        importance_weights = base_weights * kappas[i]
        
        weight_ratio = kappas[i]
        effective_sample_size = (tf.reduce_sum(importance_weights)**2) / tf.reduce_sum(importance_weights**2)
        
        if verbose:
            print(f"Region {i}: kappa = {kappas[i]:.4f}, eff_sample_size = {effective_sample_size:.0f}/20000")
    
    # For good importance sampling, effective sample size should be reasonably large
    total_weights = []
    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    for i in range(len(kappas)):
        start_idx = num_pts_per_region * i
        end_idx = num_pts_per_region * (i + 1)
        region_weights = aux_weights[start_idx:end_idx] * kappas[i]
        total_weights.extend(region_weights)
    
    total_weights = np.array(total_weights)
    eff_sample_size = (np.sum(total_weights)**2) / np.sum(total_weights**2)
    if verbose:
        print(f"Overall effective sample size: {eff_sample_size:.0f} out of {len(pts)}")
        print(f"Efficiency: {eff_sample_size/len(pts):.3f}")
        print("If efficiency < 0.1, importance sampling is probably hurting more than helping")

    return regional_variances
    
def integrate_variance_kappa_weighted(integrand, pts, wo, comp_model, variances=None, kappas=[1.], normalize_to_vol=None):
    if variances is not None:
        inv_var_weights = [1.0 / v for v in variances]
        total_inv_var_weight = sum(inv_var_weights)
        variance_weighting = [w / total_inv_var_weight for w in inv_var_weights]
    else:
        variance_weighting = [1.] * len(kappas)
        
    num_pts_per_region = len(pts) // len(kappas)
    
    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    aux_weights_weighted = []
    
    for i in range(len(kappas)):
        start_idx = num_pts_per_region * i
        end_idx = num_pts_per_region * (i + 1)
        region_weights = aux_weights[start_idx:end_idx] * kappas[i] * variance_weighting[i]
        aux_weights_weighted.append(region_weights)
    
    # Concatenate and normalize
    aux_weights_weighted = tf.concat(aux_weights_weighted, axis=0)

    res = (-1.j/2)**comp_model.nfold * tf.reduce_mean(integrand * aux_weights_weighted, axis=-1)
    if normalize_to_vol is not None:
        vol = tf.abs(tf.reduce_mean(tf.linalg.det(comp_model(pts)) * aux_weights_weighted, axis=-1))
        kappa = normalize_to_vol / vol
        vol *= tf.cast(kappa, dtype=vol.dtype)
        res *= tf.cast(kappa, dtype=res.dtype)
    return res

def config_matrix_from_metadata(metadata):
    """
    Build CICY configuration matrix from metadata['exponents'] and metadata['dimPs'].

    Returns
    -------
    ambient : list[int]
        Ambient projective dimensions, e.g. [4] or [2,2]
    config : list[list[int]]
        Configuration matrix rows (one row per equation), e.g. [[5]] or [[3,3]]
    """
    ambient = [int(x) for x in metadata["dimPs"]]
    exponents = metadata["exponents"]

    block_sizes = [n + 1 for n in ambient]
    config = []

    for eq_idx, eq_monomials in enumerate(exponents):
        eq_monomials = np.asarray(eq_monomials, dtype=int)

        if eq_monomials.ndim != 2:
            raise ValueError(f"Equation {eq_idx} exponents must be 2D, got shape {eq_monomials.shape}")

        expected_ncoords = sum(block_sizes)
        if eq_monomials.shape[1] != expected_ncoords:
            raise ValueError(
                f"Equation {eq_idx} has {eq_monomials.shape[1]} coords, expected {expected_ncoords}"
            )

        monomial_multidegrees = []
        for mono in eq_monomials:
            degs = []
            start = 0
            for bs in block_sizes:
                degs.append(int(np.sum(mono[start:start+bs])))
                start += bs
            monomial_multidegrees.append(degs)

        first = monomial_multidegrees[0]
        for d in monomial_multidegrees[1:]:
            if d != first:
                raise ValueError(
                    f"Equation {eq_idx} is not homogeneous in ambient blocks.\n"
                    f"Found multidegrees: {monomial_multidegrees}"
                )

        config.append(first)

    return ambient, config

def euler_cicy(dims, multidegrees, check_cy=False):
    """
    Compute the Euler characteristic of a complete intersection in a product of projective spaces.

    Parameters
    ----------
    dims : list of int
        Dimensions of the projective factors, e.g. [2, 2] for P^2 x P^2.
    multidegrees : list of list of int
        List of multidegrees of the defining equations.
        Each entry is a list of length len(dims).
        Example: [[3, 3]] for one equation of degree (3,3) in P^2 x P^2.
    check_cy : bool, optional
        If True, check the Calabi–Yau condition sum_j q_ij = n_i + 1 for each i.

    Returns
    -------
    sympy Integer
        The Euler characteristic χ(X).
    """
    dims = list(dims)
    multidegrees = [list(row) for row in multidegrees]
    r = len(dims)
    if any(len(row) != r for row in multidegrees):
        raise ValueError("Each multidegree must have the same length as 'dims'.")
    
    K = len(multidegrees)
    m = sum(dims)              # dim of ambient A
    d = m - K                  # dim of X = complete intersection
    
    if d < 0:
        raise ValueError("Negative dimension; check dims and multidegrees.")
    
    if check_cy:
        for i, n in enumerate(dims):
            s = sum(row[i] for row in multidegrees)
            if s != n + 1:
                raise ValueError(
                    f"Calabi–Yau condition fails for factor {i}: "
                    f"sum of degrees = {s}, expected {n+1}"
                )

    # Formal parameter for total degree bookkeeping
    t = sp.Symbol('t')
    Js = sp.symbols('J0:'+str(r))   # J0, J1, ..., J_{r-1}

    # Total Chern class of the ambient: c(TA) = Π_i (1 + J_i)^{n_i+1}
    cTA_t = 1
    for i, n in enumerate(dims):
        cTA_t *= (1 + t*Js[i])**(n + 1)

    # Total Chern class of the normal bundle: c(N) = Π_j (1 + D_j)
    # where D_j = Σ_i q_ij J_i
    cN_t = 1
    for row in multidegrees:
        D = sum(row[i]*Js[i] for i in range(r))
        cN_t *= (1 + t*D)

    # c(TX) = c(TA) / c(N), expanded as a series in t up to order d
    frac = cTA_t / cN_t
    cTX_series = sp.series(frac, t, 0, d+1).removeO()
    cTX_series = sp.expand(cTX_series)

    # c_d(TX) is the coefficient of t^d
    cd = cTX_series.coeff(t, d)

    # Class of X in the ambient: [X] = Π_j D_j
    classX = 1
    for row in multidegrees:
        D = sum(row[i]*Js[i] for i in range(r))
        classX *= D

    # Integrand for Euler number: c_d(TX) ∧ [X]
    integrand = sp.expand(cd * classX)

    # Integrate over A: pick coefficient of J0^{n0} J1^{n1} ... Jr^{nr}
    poly = sp.Poly(integrand, *Js)
    total = 0
    for monom, coeff in poly.terms():
        if all(exp == dims[i] for i, exp in enumerate(monom)):
            total += coeff

    return sp.simplify(total)
