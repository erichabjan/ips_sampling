import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

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