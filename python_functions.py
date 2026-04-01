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
from typing import List, Optional, Union
from pathlib import Path
import json

from cymetric.models.models import PhiFSModel
from cymetric.models.helper import prepare_basis as prepare_tf_basis
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator

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

def _reconstruct_coefficients_from_metadata(metadata: dict) -> List[np.ndarray]:
    coeffs = []
    for eq_coeffs in metadata["coefficients_realimag"]:
        c = np.array([complex(x["re"], x["im"]) for x in eq_coeffs], dtype=np.complex128)
        coeffs.append(c)
    return coeffs


def _ensure_int_array_nested(x) -> List[np.ndarray]:
    # metadata["exponents"] is nested: [eq][monomial][coord]
    return [np.array(eq, dtype=np.int64) for eq in x]


def compute_metric_measure_scores(
    pts, wo, h_blocks_per_region, BASIS, alpha, deg, monomials,
    kappas, j_elim=None, batch_size=512
):
    """
    For every point x_i and every region metric r, compute

        w_r(x_i) = Re(|Omega|^2 / det(g_r(x_i)))

    and the ownership score

        score_r(x_i) = |kappa_r * w_r(x_i) - 1|.

    This mirrors the Mathematica ownership logic used in
    attachMetricScoresToPointsCICY.

    Parameters
    ----------
    pts : array [N, 2*ncoords]
        Real-packed points.
    wo : array [N,2]
        wo[:,1] = |Omega|^2.
    h_blocks_per_region : dict
        region_id -> list of Hermitian blocks h = L^dag L.
    BASIS, alpha, deg, monomials
        Same arguments used to build the deformed FS models.
    kappas : array-like [n_regions]
        One kappa per region.
    j_elim : array [N, nhyper] or None
        Eliminated coordinates.
    batch_size : int
        Batch size for metric evaluation.

    Returns
    -------
    scores : np.ndarray, shape (n_regions, N)
        Ownership scores.
    metric_weights : np.ndarray, shape (n_regions, N)
        w_r(x_i) = Re(|Omega|^2 / det(g_r(x_i))).
    """
    pts = tf.cast(pts, dtype=tf.float32)
    omegas = np.asarray(wo[:, 1], dtype=np.float64).reshape(-1)
    kappas = np.asarray(kappas, dtype=np.float64).reshape(-1)

    n_regions = len(kappas)
    N = len(omegas)

    scores = np.full((n_regions, N), np.inf, dtype=np.float64)
    metric_weights = np.full((n_regions, N), np.nan, dtype=np.float64)

    for r in tqdm(range(n_regions), desc="Ownership scores"):
        h_blocks = h_blocks_per_region.get(r, None)
        model_r = DeformedFSModelComp(
            None, BASIS=BASIS, alpha=alpha,
            deg=deg, monomials=monomials,
            h_blocks=h_blocks
        )

        num_batches = math.ceil(N / batch_size)
        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, N)

            pts_b = pts[s:e]
            jel_b = None
            if j_elim is not None:
                jel_b = tf.convert_to_tensor(np.asarray(j_elim)[s:e], dtype=tf.int64)

            g_b = model_r(pts_b, j_elim=jel_b).numpy().astype(np.complex128)
            detg_b = np.linalg.det(g_b)

            # Mathematica logic: w = Re[OmegaOmegaBar / detgNorm]
            w_b = np.real(omegas[s:e] / detg_b)

            bad = ~np.isfinite(w_b)
            score_b = np.abs(kappas[r] * w_b - 1.0)
            score_b[bad] = np.inf

            metric_weights[r, s:e] = w_b
            scores[r, s:e] = score_b

    return scores, metric_weights


def compute_region_ownership_labels(
    pts, wo, h_blocks_per_region, BASIS, alpha, deg, monomials,
    kappas, j_elim=None, batch_size=512, ownership_tol=0.0,
    return_details=False
):
    """
    Compute geometric ownership labels in Python.

    The owner of a point is the region metric r minimizing

        |kappa_r * w_r(x) - 1|,

    matching the Mathematica score.

    Parameters
    ----------
    ownership_tol : float
        Optional tie tolerance. If > 0, points within ownership_tol of the
        minimum score are still assigned to the first minimum.

    Returns
    -------
    owner_labels : np.ndarray, shape (N,)
        0-indexed owner region of each point.
    If return_details:
        owner_labels, min_scores, scores, metric_weights
    """
    scores, metric_weights = compute_metric_measure_scores(
        pts, wo, h_blocks_per_region, BASIS, alpha, deg, monomials,
        kappas, j_elim=j_elim, batch_size=batch_size
    )

    min_scores = np.min(scores, axis=0)
    owner_labels = np.argmin(scores, axis=0).astype(np.int64)

    if ownership_tol > 0:
        # keep first minimum among all nearly-tied regions
        for i in range(scores.shape[1]):
            near = np.where(scores[:, i] <= min_scores[i] + ownership_tol)[0]
            owner_labels[i] = int(near[0])

    if return_details:
        return owner_labels, min_scores, scores, metric_weights
    return owner_labels

def estimate_region_volume_fractions(
    wo, source_region_labels, owner_labels, kappas
):
    """
    Estimate region volume fractions f_r from ownership labels.

    The source sampling distribution is corrected using the same IPS
    Kähler-volume weights that appear in the top-form estimator:

        W_i = (wo[i,0] / wo[i,1]) * kappa_source(i)

    Then the volume fraction of ownership region r is estimated by

        f_r = sum_{i owned by r} W_i / sum_i W_i.

    Parameters
    ----------
    wo : array [N,2]
        wo[:,0] = IPS weights, wo[:,1] = |Omega|^2
    source_region_labels : array [N]
        0-indexed region labels indicating which metric generated each point.
    owner_labels : array [N]
        0-indexed ownership labels.
    kappas : array [n_regions]
        One kappa per source region.

    Returns
    -------
    f : np.ndarray, shape (n_regions,)
        Estimated region volume fractions, summing to ~1.
    point_weights : np.ndarray, shape (N,)
        The per-point source-corrected Kähler-volume weights W_i.
    """
    source_region_labels = np.asarray(source_region_labels, dtype=np.int64).reshape(-1)
    owner_labels = np.asarray(owner_labels, dtype=np.int64).reshape(-1)
    kappas = np.asarray(kappas, dtype=np.float64).reshape(-1)

    if len(source_region_labels) != len(owner_labels):
        raise ValueError("source_region_labels and owner_labels must have the same length")

    aux_weights = np.asarray(wo[:, 0] / wo[:, 1], dtype=np.float64).reshape(-1)
    if len(aux_weights) != len(owner_labels):
        raise ValueError("wo must have the same number of rows as owner_labels")

    kappa_source = kappas[source_region_labels]
    point_weights = aux_weights * kappa_source

    total = np.sum(point_weights)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Total source-corrected weight is nonpositive or nonfinite")

    n_regions = len(kappas)
    numerators = np.bincount(owner_labels, weights=point_weights, minlength=n_regions)
    f = numerators / total

    return f, point_weights

def cy_volume_from_intersections(
    input_dir: Union[str, Path],
    num_regions: int = 1,
    kahler_moduli: Optional[np.ndarray] = None,
    metadata_filename: Optional[str] = None,
) -> float:
    """
    Compute the Kähler volume Vol_K = d_{ijk} t_i t_j t_k for a CICY 3-fold,
    using intersection numbers computed by cymetric from the geometry saved
    in metadata_{num_regions}.json.

    Returns the intersection-number volume in cymetric's convention.
    """
    input_dir = Path(input_dir)
    if metadata_filename is None:
        metadata_filename = f"metadata_{num_regions}.json"
    meta_path = input_dir / metadata_filename

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Geometry
    ambient = np.array(metadata.get("ambient", metadata["dimPs"]), dtype=np.int64)
    monomials = _ensure_int_array_nested(metadata["exponents"])
    coefficients = _reconstruct_coefficients_from_metadata(metadata)

    # Kähler moduli
    if kahler_moduli is None:
        if "kahler_moduli" in metadata:
            kahler_moduli = np.array(metadata["kahler_moduli"], dtype=np.float64)
        else:
            kahler_moduli = np.ones(len(ambient), dtype=np.float64)
    else:
        kahler_moduli = np.array(kahler_moduli, dtype=np.float64)

    if kahler_moduli.shape != (len(ambient),):
        raise ValueError(
            "kahler_moduli must have shape ({},), got {}".format(len(ambient), kahler_moduli.shape)
        )

    # Instantiate point generator (we only need intersection tensor)
    pg = CICYPointGenerator(monomials, coefficients, kahler_moduli, ambient)

    nfold = int(pg.nfold)
    if nfold != 3:
        raise ValueError("This function is for CY 3-folds; got nfold={}".format(nfold))

    d = pg.intersection_tensor  # shape (h11,h11,h11)
    vol = float(np.einsum("ijk,i,j,k->", d, kahler_moduli, kahler_moduli, kahler_moduli))
    return vol

def christoffel(g_model, pts, j_elim=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = g_model(pts, j_elim=j_elim)
        g_re, g_im = tf.math.real(g), tf.math.imag(g)
    d_g_re = tape.batch_jacobian(g_re, pts, experimental_use_pfor=False)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = g_model(pts, j_elim=j_elim)
        g_im = tf.math.imag(g)
    d_g_im = tape.batch_jacobian(g_im, pts, experimental_use_pfor=False)

    dx_g_re, dy_g_re = d_g_re[:,:,:,:g_model.ncoords], d_g_re[:,:,:,g_model.ncoords:]
    dx_g_im, dy_g_im = d_g_im[:,:,:,:g_model.ncoords], d_g_im[:,:,:,g_model.ncoords:]
    d_g = tf.complex(dx_g_re + dy_g_im, -dy_g_re + dx_g_im)

    pbs = g_model.pullbacks(pts, j_elim=j_elim)
    d_g_pb = tf.einsum('xbn,xcdn->xbcd', pbs, d_g)
    gamma = tf.einsum('xbcd, xda->xabc', d_g_pb, tf.linalg.inv(g))
    return gamma

def riemann(g_model, pts, j_elim=None):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts, j_elim=j_elim)
        gamma_re = tf.math.real(gamma)
    d_gamma_re = tape.batch_jacobian(gamma_re, pts, experimental_use_pfor=False)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        gamma = christoffel(g_model, pts, j_elim=j_elim)
        gamma_im = tf.math.imag(gamma)
    d_gamma_im = tape.batch_jacobian(gamma_im, pts, experimental_use_pfor=False)

    dx_gamma_re, dy_gamma_re = d_gamma_re[:,:,:,:,:g_model.ncoords], d_gamma_re[:,:,:,:,g_model.ncoords:]
    dx_gamma_im, dy_gamma_im = d_gamma_im[:,:,:,:,:g_model.ncoords], d_gamma_im[:,:,:,:,g_model.ncoords:]
    dbarc_gamma = tf.complex(dx_gamma_re - dy_gamma_im, dy_gamma_re + dx_gamma_im)

    pbs = g_model.pullbacks(pts, j_elim=j_elim)
    riem = -tf.einsum('xci,xabdi->xabcd', tf.math.conj(pbs), dbarc_gamma)
    return riem

# compute Ric from \del \bar\del \log \det g
def get_Ricci(g_model, pts, j_elim=None):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(pts)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(pts)
            ldg = tf.math.log(tf.linalg.det(g_model(pts, j_elim=j_elim)))
        d_ldg = tape2.gradient(ldg, pts)
    dd_ldg = tape1.batch_jacobian(d_ldg, pts, experimental_use_pfor=False)

    dx_dx_ldg, dx_dy_ldg = 0.25*dd_ldg[:, :g_model.ncoords, :g_model.ncoords], 0.25*dd_ldg[:, :g_model.ncoords, g_model.ncoords:]
    dy_dx_ldg, dy_dy_ldg = 0.25*dd_ldg[:, g_model.ncoords:, :g_model.ncoords], 0.25*dd_ldg[:, g_model.ncoords:, g_model.ncoords:]
    dd_ldg = tf.complex(dx_dx_ldg + dy_dy_ldg, dx_dy_ldg - dy_dx_ldg)

    pbs = g_model.pullbacks(pts, j_elim=j_elim)
    dd_ldg = tf.einsum('xai,xij,xbj->xab', pbs, dd_ldg, tf.math.conj(pbs))
    return dd_ldg


## Definition for kappa- and variance-weighted integrations
def compute_riemann(target_file, pts, comp_model, j_elim=None, batch_size=10000):
    pts = tf.cast(pts, dtype=tf.float32)
    num_batches = math.ceil(len(pts) / batch_size)

    if not os.path.exists(target_file):
        for i in tqdm(range(num_batches)):
            s = i * batch_size
            e = min((i + 1) * batch_size, len(pts))
            jel = None
            if j_elim is not None:
                jel = tf.convert_to_tensor(j_elim[s:e], dtype=tf.int64)

            chunk = riemann(comp_model, pts[s:e], j_elim=jel)
            riem = chunk if i == 0 else tf.concat([riem, chunk], axis=0)

        with open(target_file, 'wb') as hnd:
            pickle.dump(riem.numpy(), hnd)
    else:
        with open(target_file, 'rb') as hnd:
            riem = pickle.load(hnd)

    return tf.cast(riem, dtype=tf.complex64)

def get_chern_classes(riem, comp_model, g_cy=None):
    """
    Compute Chern classes using the original contraction convention that was
    already producing reasonable Euler estimates in the single-region code.

    If g_cy is provided, also return
        chi_density = c3_form / omega_top
    for diagnostics / future regional-volume estimators.

    Returns
    -------
    If g_cy is None:
        c1, c2, c3, c3_form
    If g_cy is not None:
        c1, c2, c3, c3_form, chi_density
    """
    # first Chern class
    tr_R = -tf.einsum('xaabc->xcb', riem)
    c1 = 1.j / (2 * math.pi) * tr_R

    # second Chern class
    tr_R2 = tf.einsum('xabmn, xbaop->xnmpo', riem, riem)
    c2 = 1.0 / (2 * (2 * math.pi) ** 2) * (
        tr_R2 - tf.einsum('xab,xcd->xabcd', tr_R, tr_R)
    )

    # third Chern class
    tr_R3 = tf.einsum('xabmn, xbcop, xcaqr->xnmporq', riem, riem, riem)
    c3 = (
        1.0 / 3.0 * tf.einsum('xmn,xopqr->xmnopqr', c1, c2)
        + 1.0 / (3 * (2 * math.pi) ** 2) * tf.einsum('xmn,xopqr->xmnopqr', c1, tr_R2)
        - 1.j / (3 * (2 * math.pi) ** 3) * tr_R3
    )

    c3_form = 1.0 / math.factorial(comp_model.nfold) * tf.einsum(
        'xmnopqr,moq,npr->x', c3, comp_model.lc, comp_model.lc
    )

    if g_cy is not None:
        omega_top = compute_kahler_top_contraction(
            g_cy, comp_model.lc, comp_model.nfold
        )
        chi_density = c3_form / tf.cast(omega_top, c3_form.dtype)
        return c1, c2, c3, c3_form, chi_density

    return c1, c2, c3, c3_form


def compute_kahler_top_contraction(g_cy, lc, nfold):
    """Contract the Kähler top form omega^n / n! with the same epsilon pattern
    used for c3_form.

    Convention:
        omega_{a \bar b} = (i/2) g_{a \bar b}

    With the same LC contraction pattern as c3_form, this should satisfy
    pointwise:
        omega_top / det(g) = (i/2)^n
    up to numerical error.
    """
    lc128 = tf.cast(lc, tf.complex128)
    g128 = tf.cast(g_cy, tf.complex128)

    # Kähler 2-form in the same real-volume convention used by the old
    # (-i/2)^n prefactor machinery.
    omega = tf.constant(0.5j, dtype=tf.complex128) * g128

    if nfold == 3:
        omega_top = (1.0 / math.factorial(3)) * tf.einsum(
            'xmn,xop,xqr,moq,npr->x', omega, omega, omega, lc128, lc128
        )
    elif nfold == 2:
        omega_top = (1.0 / math.factorial(2)) * tf.einsum(
            'xmn,xop,mo,np->x', omega, omega, lc128, lc128
        )
    elif nfold == 1:
        omega_top = omega[:, 0, 0]
    else:
        raise ValueError(f"compute_kahler_top_contraction: unsupported nfold={nfold}")

    return omega_top

def kahler_top_diagnostics(g_cy, lc, nfold):
    """Diagnostic for the Kähler top-form normalization.

    Returns
    -------
    omega_top : complex128 [N]
        Contracted Kähler top-form scalar.
    det_g : complex128 [N]
        Determinant of the CY metric.
    ratio : complex128 [N]
        omega_top / det(g), which should be approximately (i/2)^n pointwise.
    """
    g128 = tf.cast(g_cy, tf.complex128)
    omega_top = compute_kahler_top_contraction(g_cy, lc, nfold)
    det_g = tf.linalg.det(g128)
    ratio = omega_top / det_g
    return omega_top, det_g, ratio


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
    
def analyze_regions(integrand, pts, wo, comp_model, kappas, region_labels, verbose=True):
    region_labels = np.asarray(region_labels).reshape(-1).astype(int)
    unique_regions = np.sort(np.unique(region_labels))

    if len(unique_regions) != len(kappas):
        raise ValueError(
            f"Found {len(unique_regions)} unique regions in region_labels, "
            f"but {len(kappas)} kappas."
        )

    if verbose:
        print("=== Kappa Analysis ===")
        print(f"Kappa values: {kappas}")
        print(f"Kappa std/mean: {np.std(kappas)/np.mean(kappas):.3f}")

    if verbose:
        print("\n=== Regional Contribution Analysis ===")

    regional_contributions = []
    regional_variances = []

    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)

    for r in unique_regions:
        mask = (region_labels == r)

        region_integrand = tf.boolean_mask(integrand, mask)
        region_weights = tf.boolean_mask(aux_weights, mask)

        raw_contribution = tf.reduce_mean(region_integrand * region_weights)
        integrand_var = tf.reduce_mean(
            tf.abs(region_integrand - tf.reduce_mean(region_integrand))**2
        )

        regional_contributions.append(raw_contribution.numpy())
        regional_variances.append(float(integrand_var.numpy()))

        if verbose:
            print(
                f"Region {r}: N = {np.count_nonzero(mask)}, "
                f"contribution = {raw_contribution:.3f}, variance = {integrand_var:.6f}"
            )

    if verbose:
        print(f"Regional contribution std: {np.std(regional_contributions):.3f}")

    total_weights = []
    for i, r in enumerate(unique_regions):
        mask = (region_labels == r)
        region_weights = aux_weights.numpy()[mask] * kappas[i]
        total_weights.extend(region_weights)

    total_weights = np.asarray(total_weights)
    eff_sample_size = (np.sum(total_weights)**2) / np.sum(total_weights**2)

    if verbose:
        print(f"Overall effective sample size: {eff_sample_size:.0f} out of {len(pts)}")
        print(f"Efficiency: {eff_sample_size/len(pts):.3f}")

    return regional_variances
    
def integrate_variance_kappa_weighted(
    integrand, pts, wo, comp_model, region_labels,
    variances=None, kappas=[1.], normalize_to_vol=None
):
    region_labels = np.asarray(region_labels).reshape(-1).astype(int)
    unique_regions = np.sort(np.unique(region_labels))

    if len(unique_regions) != len(kappas):
        raise ValueError(
            f"Found {len(unique_regions)} unique regions in region_labels, "
            f"but {len(kappas)} kappas."
        )

    if variances is not None:
        inv_var_weights = [1.0 / v for v in variances]
        total_inv_var_weight = sum(inv_var_weights)
        variance_weighting = [w / total_inv_var_weight for w in inv_var_weights]
    else:
        variance_weighting = [1.0] * len(kappas)

    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
    aux_weights_weighted = tf.TensorArray(tf.complex64, size=len(aux_weights))

    for i, r in enumerate(unique_regions):
        mask = (region_labels == r)
        idx = np.where(mask)[0]

        region_weights = aux_weights.numpy()[idx] * kappas[i] * variance_weighting[i]
        for j, ind in enumerate(idx):
            aux_weights_weighted = aux_weights_weighted.write(ind, region_weights[j])

    aux_weights_weighted = aux_weights_weighted.stack()

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


###############################################################################
# IPS-consistent metric, curvature, and integration functions
###############################################################################

def h_blocks_from_L(L_matrices):
    """Compute Hermitian deformation blocks h_i = L_i^dag L_i from L matrices.

    Args:
        L_matrices: list of numpy arrays, one per projective factor.
                    Each L_i has shape (n_i+1, n_i+1).

    Returns:
        list of numpy arrays h_i = L_i^dag @ L_i (positive-definite Hermitian).
    """
    return [np.conj(L.T) @ L for L in L_matrices]


def _deformed_fs_block(z, h, t):
    """Deformed Fubini-Study metric on one P^n factor.

    Computes  g_{a bar{b}} = (t / pi) (s H^T_{ab} - conj(Hz)_a (Hz)_b) / s^2
    where  s = bar{z} . h . z  and  Hz = h . z.

    This matches the Mathematica ``FSBlockMetric`` convention and reduces
    to the standard cymetric FS metric when h = I.

    Args:
        z: tf.complex64 [batch, n+1] — homogeneous coordinates for this P^n.
        h: tf.complex64 [n+1, n+1] — Hermitian matrix (h = L^dag L).
        t: tf.complex64 scalar — Kähler modulus for this factor.

    Returns:
        tf.complex64 [batch, n+1, n+1] — metric in ambient coordinates.
    """
    pi_c = tf.constant(np.pi, dtype=tf.complex64)
    hz = tf.einsum('ij,xj->xi', h, z)                         # H·z  [batch, n+1]
    conj_hz = tf.math.conj(hz)                                # conj(H·z) = z̄·H  [batch, n+1]
    zhz = tf.reduce_sum(tf.math.conj(z) * hz, axis=-1)        # z̄·h·z  [batch]

    outer = tf.einsum('xa,xb->xab', conj_hz, hz)              # conj(Hz)_a (Hz)_b
    h_T = tf.transpose(h)                                     # H^T = conj(H) for Hermitian
    h_T_broad = tf.broadcast_to(h_T[None, :, :],
                                [tf.shape(z)[0], tf.shape(h_T)[0], tf.shape(h_T)[1]])

    numerator = tf.einsum('x,xab->xab', zhz, h_T_broad) - outer
    zhz_sq = tf.reshape(zhz * zhz, [-1, 1, 1])

    return (t / pi_c) * numerator / zhz_sq


class DeformedFSModelComp(SpectralFSModelComp):
    """SpectralFSModelComp whose FS metric is deformed by per-block Hermitian
    matrices  h_i = L_i^dag L_i.   When h_blocks is None the standard FS
    metric is used (equivalent to h_i = I for all i).

    Usage::

        model = DeformedFSModelComp(
            None, BASIS=BASIS, alpha=alpha, deg=2, monomials=monomials,
            h_blocks=[h1, h2]     # one Hermitian per projective factor
        )
        g_cy = model(pts, j_elim=jel)   # deformed FS pulled back to CY
    """

    def __init__(self, *args, h_blocks=None, **kwargs):
        super().__init__(*args, **kwargs)
        if h_blocks is not None:
            self.h_blocks = [
                tf.cast(tf.constant(np.asarray(h, dtype=np.complex64)),
                        dtype=tf.complex64)
                for h in h_blocks
            ]
        else:
            self.h_blocks = None

    def call(self, input_tensor, training=True, j_elim=None):
        if self.h_blocks is None:
            return self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return self._deformed_fubini_study_pb(input_tensor, j_elim=j_elim)

    def _deformed_fubini_study_pb(self, points, j_elim=None):
        ts = self.BASIS['KMODULI']
        pb = self.pullbacks(points, j_elim=j_elim)

        if self.nProjective > 1:
            cpoints = tf.complex(
                points[:, :self.degrees[0]],
                points[:, self.ncoords:self.ncoords + self.degrees[0]])
            fs = _deformed_fs_block(cpoints, self.h_blocks[0], ts[0])
            fs = tf.einsum('xij,ia,bj->xab', fs,
                           self.proj_matrix['0'],
                           tf.transpose(self.proj_matrix['0']))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e],
                                     points[:, self.ncoords + s:
                                                self.ncoords + e])
                fs_tmp = _deformed_fs_block(cpoints, self.h_blocks[i], ts[i])
                fs_tmp = tf.einsum('xij,ia,bj->xab', fs_tmp,
                                   self.proj_matrix[str(i)],
                                   tf.transpose(self.proj_matrix[str(i)]))
                fs += fs_tmp
        else:
            cpoints = tf.complex(
                points[:, :self.ncoords],
                points[:, self.ncoords:2 * self.ncoords])
            fs = _deformed_fs_block(cpoints, self.h_blocks[0], ts[0])

        return tf.einsum('xai,xij,xbj->xab', pb, fs, tf.math.conj(pb))


# ── optimised curvature (stacked real/imag → single tape per level) ──────

def christoffel_opt(g_model, pts, j_elim=None):
    """Christoffel symbols Gamma^a_{bc} with a single GradientTape call,
    keeping the full cymetric pipeline in its native float32/complex64 dtype.
    """
    ncoords = g_model.ncoords

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        g = g_model(pts, j_elim=j_elim)   # complex64
        g_stacked = tf.concat([tf.math.real(g), tf.math.imag(g)], axis=1)  # float32

    d_g_stacked = tape.batch_jacobian(
        g_stacked, pts, experimental_use_pfor=False
    )
    del tape

    nf = int(g.shape[1])
    d_g_re = d_g_stacked[:, :nf, :, :]
    d_g_im = d_g_stacked[:, nf:, :, :]

    dx_g_re, dy_g_re = d_g_re[:, :, :, :ncoords], d_g_re[:, :, :, ncoords:]
    dx_g_im, dy_g_im = d_g_im[:, :, :, :ncoords], d_g_im[:, :, :, ncoords:]

    d_g = tf.complex(dx_g_re + dy_g_im, -dy_g_re + dx_g_im)  # complex64

    pbs = g_model.pullbacks(pts, j_elim=j_elim)  # complex64
    d_g_pb = tf.einsum('xbn,xcdn->xbcd', pbs, d_g)
    gamma = tf.einsum('xbcd,xda->xabc', d_g_pb, tf.linalg.inv(g))

    return gamma


def riemann_opt(g_model, pts, j_elim=None):
    """Riemann tensor R^a_{b,c bar{d}} with a single GradientTape call
    at each differentiation level, using native float32/complex64 dtype.
    """
    ncoords = g_model.ncoords

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        gamma = christoffel_opt(g_model, pts, j_elim=j_elim)   # complex64
        gamma_stacked = tf.concat(
            [tf.math.real(gamma), tf.math.imag(gamma)], axis=1
        )  # float32

    d_gamma_stacked = tape.batch_jacobian(
        gamma_stacked, pts, experimental_use_pfor=False
    )
    del tape

    nf = int(gamma.shape[1])
    d_re = d_gamma_stacked[:, :nf, :, :, :]
    d_im = d_gamma_stacked[:, nf:, :, :, :]

    dx_re, dy_re = d_re[:, :, :, :, :ncoords], d_re[:, :, :, :, ncoords:]
    dx_im, dy_im = d_im[:, :, :, :, :ncoords], d_im[:, :, :, :, ncoords:]

    dbarc_gamma = tf.complex(dx_re - dy_im, dy_re + dx_im)  # complex64

    pbs = g_model.pullbacks(pts, j_elim=j_elim)  # complex64
    return -tf.einsum('xci,xabdi->xabcd', tf.math.conj(pbs), dbarc_gamma)


# ── per-region Riemann computation ───────────────────────────────────────

def compute_riemann_regional(
    pts, region_labels, h_blocks_per_region,
    BASIS, alpha, deg, monomials,
    j_elim=None, batch_size=512, return_metric=False
):
    """Compute the Riemann tensor for every point using the deformed FS
    metric of that point's region.

    Args:
        pts:  np or tf array [N, 2*ncoords], float.
        region_labels:  int array [N] (0-indexed).
        h_blocks_per_region:  dict  {region_id: list_of_h_matrices} where
            each list has one Hermitian h_i per projective factor.
            Region 0 may use ``None`` or a list of identity matrices.
        BASIS, alpha, deg, monomials:  same arguments used to construct
            ``SpectralFSModelComp``.
        j_elim:  int array [N, nhyper] or None.
        batch_size:  batch size for Riemann computation.
        return_metric:  if True, also return the CY metric g at each point,
            evaluated with the same regional model used for curvature.

    Returns:
        tf.complex64 tensor [N, nfold, nfold, nfold, nfold]
        when return_metric is False.

        (riem, g_cy) tuple when return_metric is True, where g_cy is
        tf.complex64 tensor [N, nfold, nfold].
    """
    pts = tf.cast(pts, dtype=tf.float32)
    region_labels = np.asarray(region_labels).reshape(-1).astype(int)
    unique_regions = np.sort(np.unique(region_labels))

    # determine output shape from a throwaway model
    tmp = SpectralFSModelComp(None, BASIS=BASIS, alpha=alpha,
                              deg=deg, monomials=monomials)
    nf = tmp.nfold
    del tmp

    N = len(pts)
    riem_out = np.zeros((N, nf, nf, nf, nf), dtype=np.complex64)
    if return_metric:
        g_out = np.zeros((N, nf, nf), dtype=np.complex64)

    for r in tqdm(unique_regions, desc="Riemann per region"):
        mask = (region_labels == r)
        idx = np.where(mask)[0]
        pts_r = tf.gather(pts, idx)
        jel_r = None
        if j_elim is not None:
            jel_r = tf.convert_to_tensor(
                np.asarray(j_elim)[idx], dtype=tf.int64)

        h_blocks = h_blocks_per_region.get(r, None)
        model_r = DeformedFSModelComp(
            None, BASIS=BASIS, alpha=alpha,
            deg=deg, monomials=monomials,
            h_blocks=h_blocks
        )

        num_batches = math.ceil(len(pts_r) / batch_size)
        riem_chunks = []
        g_chunks = [] if return_metric else None
        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, len(pts_r))
            jb = jel_r[s:e] if jel_r is not None else None
            riem_chunks.append(riemann_opt(model_r, pts_r[s:e], j_elim=jb).numpy())
            if return_metric:
                g_chunks.append(model_r(pts_r[s:e], j_elim=jb).numpy())

        riem_out[idx] = np.concatenate(riem_chunks, axis=0)
        if return_metric:
            g_out[idx] = np.concatenate(g_chunks, axis=0)

    riem_tensor = tf.cast(riem_out, dtype=tf.complex64)
    if return_metric:
        g_tensor = tf.cast(g_out, dtype=tf.complex64)
        return riem_tensor, g_tensor
    return riem_tensor


# ── corrected IPS integration ────────────────────────────────────────────

def integrate_euler_top_form_regional(
    integrand, pts, wo, comp_model, region_labels, kappas, normalize_to_vol=None
):
    """
    Generalization of the old working top-form estimator.

    This reduces to the old single-region estimator when there is only one
    region, but also supports per-point regional kappas.

    Parameters
    ----------
    integrand : complex tensor [N]
        Typically c3_form.
    pts : tensor/array [N, 2*ncoords]
        Real-packed CY points.
    wo : array [N,2]
        wo[:,0] = IPS weights
        wo[:,1] = |Omega|^2
    comp_model : model
        Used for nfold and for sampled-volume normalization.
    region_labels : array [N]
        0-indexed region labels.
    kappas : array [n_regions]
        One kappa per region.
    normalize_to_vol : float or None
        If provided, rescale using the sampled Kähler volume as in the old code.

    Returns
    -------
    complex scalar
    """
    region_labels = np.asarray(region_labels).reshape(-1).astype(int)
    unique_regions = np.sort(np.unique(region_labels))

    if len(unique_regions) != len(kappas):
        raise ValueError(
            f"Found {len(unique_regions)} unique regions in region_labels, "
            f"but {len(kappas)} kappas."
        )

    aux_weights = np.asarray(wo[:, 0] / wo[:, 1], dtype=np.complex64)

    # Per-point regional kappa
    kappa_pt = np.zeros(len(region_labels), dtype=np.float32)
    for i, r in enumerate(unique_regions):
        kappa_pt[region_labels == r] = kappas[i]

    total_weights = tf.convert_to_tensor(aux_weights * kappa_pt, dtype=tf.complex64)
    integrand = tf.cast(integrand, dtype=tf.complex64)

    res = (-1.j / 2) ** comp_model.nfold * tf.reduce_mean(integrand * total_weights, axis=-1)

    if normalize_to_vol is not None:
        pts_tf = tf.cast(pts, dtype=tf.float32)
        vol = tf.abs(
            tf.reduce_mean(
                tf.linalg.det(comp_model(pts_tf)) * total_weights,
                axis=-1
            )
        )
        kappa_norm = normalize_to_vol / vol
        res *= tf.cast(kappa_norm, dtype=res.dtype)

    return res

def integrate_euler_ips(
    integrand, wo, kappas, region_labels, nfold, normalize_to_vol,
    normalized=False, region_volume_weights=None
):
    """Compute the Euler characteristic from IPS samples.

    Parameters
    ----------
    integrand : array/tensor
        If normalized=True, this is chi_density defined by
            c_n = chi_density * (omega^n / n!)
        and should be averaged with respect to the Kähler volume measure.

        If normalized=False, this is the raw top-form integrand and the
        legacy IPS weighting is used.
    wo : array-like, shape (N,2)
        wo[:,0] = IPS weight, wo[:,1] = |Omega|^2
    kappas : array-like
        One kappa per region.
    region_labels : array-like
        0-indexed region id per point.
    nfold : int
        Complex dimension.
    normalize_to_vol : float
        Target Kähler volume.
    normalized : bool
        Whether integrand is chi_density.
    region_volume_weights : None or array-like
        Optional weights V_r / sum_s V_s for combining regional means when
        multiple sampling metrics are used.

    Notes
    -----
    For normalized=True, do NOT use wo[:,0]/wo[:,1]. chi_density is already
    a scalar with respect to the Kähler measure, so the correct estimator is:

        chi = Vol_K * mean(chi_density)                  (single metric)

    or, for multiple metrics / ownership regions,

        chi = Vol_K * sum_r f_r * mean_r(chi_density)

    where f_r are the region volume fractions.
    """
    integrand = tf.cast(integrand, tf.complex64)
    region_labels = np.asarray(region_labels).reshape(-1).astype(int)
    unique_regions = np.sort(np.unique(region_labels))

    if normalized:
        # Single-metric case or no supplied regional volume fractions:
        # use the ordinary sample mean of chi_density.
        if len(unique_regions) == 1 or region_volume_weights is None:
            return tf.cast(normalize_to_vol, tf.complex64) * tf.reduce_mean(integrand)

        # Multi-metric case: combine regional means with explicit region-volume weights.
        region_volume_weights = np.asarray(region_volume_weights, dtype=np.float64)
        if region_volume_weights.shape != (len(unique_regions),):
            raise ValueError(
                f"region_volume_weights must have shape ({len(unique_regions)},), "
                f"got {region_volume_weights.shape}"
            )
        if not np.isclose(np.sum(region_volume_weights), 1.0, rtol=0, atol=1e-8):
            raise ValueError("region_volume_weights must sum to 1.")

        regional_means = []
        for r in unique_regions:
            mask = (region_labels == r)
            vals_r = tf.boolean_mask(integrand, mask)
            if tf.size(vals_r) == 0:
                raise ValueError(f"Region {r} has no points.")
            regional_means.append(tf.reduce_mean(vals_r))

        regional_means = tf.stack(regional_means)
        weights_tf = tf.cast(region_volume_weights, tf.complex64)
        return tf.cast(normalize_to_vol, tf.complex64) * tf.reduce_sum(weights_tf * regional_means)

    # Legacy raw-top-form path
    aux_weights = tf.convert_to_tensor(wo[:, 0] / wo[:, 1], dtype=tf.complex64)

    kappa_arr = np.zeros(len(region_labels), dtype=np.float64)
    for i, r in enumerate(unique_regions):
        kappa_arr[region_labels == r] = kappas[i]
    kappa_pt = tf.cast(kappa_arr, dtype=tf.complex64)

    weighted = aux_weights * kappa_pt
    return (-1.j / 2) ** nfold * tf.reduce_mean(integrand * weighted, axis=-1)


def integrate_euler_density_by_owner_regions(
    chi_density, owner_labels, region_volume_fractions, point_weights, normalize_to_vol
):
    """
    Compute

        chi = Vol_K * sum_r f_r * <chi_density>_r

    where both f_r and <chi_density>_r are estimated with the same
    source-corrected Kähler-volume importance weights.

    Parameters
    ----------
    chi_density : array/tensor [N]
        Scalar Euler density c3 / (omega^n / n!).
    owner_labels : array [N]
        0-indexed ownership labels.
    region_volume_fractions : array [n_regions]
        Fractions f_r summing to 1.
    point_weights : array [N]
        Source-corrected Kähler-volume importance weights
            W_i = kappa_source(i) * wo[i,0] / wo[i,1]
    normalize_to_vol : float
        Total Kähler volume.

    Returns
    -------
    complex scalar
    """
    chi_density = np.asarray(chi_density, dtype=np.complex128).reshape(-1)
    owner_labels = np.asarray(owner_labels, dtype=np.int64).reshape(-1)
    f = np.asarray(region_volume_fractions, dtype=np.float64).reshape(-1)
    point_weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)

    if len(chi_density) != len(owner_labels):
        raise ValueError("chi_density and owner_labels must have the same length")
    if len(point_weights) != len(owner_labels):
        raise ValueError("point_weights and owner_labels must have the same length")
    if not np.isclose(np.sum(f), 1.0, rtol=0, atol=1e-8):
        raise ValueError("region_volume_fractions must sum to 1")

    regional_means = np.zeros(len(f), dtype=np.complex128)

    for r in range(len(f)):
        mask = (owner_labels == r)
        if not np.any(mask):
            regional_means[r] = 0.0 + 0.0j
            continue

        w_r = point_weights[mask]
        x_r = chi_density[mask]

        denom = np.sum(w_r)
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError(f"Nonpositive or nonfinite total weight in owner region {r}")

        regional_means[r] = np.sum(w_r * x_r) / denom

    return np.complex128(normalize_to_vol) * np.sum(f * regional_means)


def integrate_euler_multi_region_density(
    chi_density, wo, source_region_labels, owner_labels, kappas, normalize_to_vol
):
    """
    Convenience wrapper that:
      1) estimates region volume fractions from source IPS weights and owner labels
      2) computes Vol_K * sum_r f_r * <chi_density>_r
         with weighted regional means
    """
    f_r, point_weights = estimate_region_volume_fractions(
        wo=wo,
        source_region_labels=source_region_labels,
        owner_labels=owner_labels,
        kappas=kappas
    )

    chi = integrate_euler_density_by_owner_regions(
        chi_density=chi_density,
        owner_labels=owner_labels,
        region_volume_fractions=f_r,
        point_weights=point_weights,
        normalize_to_vol=normalize_to_vol
    )

    return chi, f_r, point_weights