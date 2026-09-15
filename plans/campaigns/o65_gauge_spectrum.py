#!/usr/bin/env python3
"""Gauss-Newton spectrum of the GAUGE observable, from modelled-stage dumps,
and its comparison with the mark spectrum.

  o65_gauge_spectrum.py <eps> <obs table> <base dump> <column dumps...>
        --sigma-obs S  --sigma-alpha A
        [--marks <mark base dump> <mark column dumps...>]
        [--ar1 RHO] [--demean] [--no-mask] [--sigma-scan]

Dumps are -adjoint_obs_model_dump tables (obs-table format: "ngauges K",
one line of gauge cells, then K rows "time wse_1 ... wse_ngauges"), each with
a companion <dump>.zb ("# ...", then "g cell zb" per gauge). Columns must be
named ...col<CODE>.txt. The observation table is the real one the runs used
(obs_turning_h29_41.txt); its NaNs and the above-bed rule rebuild the kept
mask exactly as the driver did (134 of 624 for the production window).

WHAT THIS COMPUTES

  S[i,k] = ( wse_i(alpha + eps e_k) - wse_i(alpha) ) / eps        (624 x 15)

over every (gauge, time) i. The Gauss-Newton Hessian of the gauge misfit in
alpha is H = S^T W S / sigma^2 with W the weighting; whitened by the prior
(sigma_alpha, uniform relative) its eigenvalues count how many roughness
combinations the GAUGES determine, exactly as o58_gauss_newton.py does for
the marks -- same construction, different rows. The gauge observable has no
argmax, so no peak-time check is needed; the self-check is instead that the
base dump's masked J reproduces the driver's printed J.

W is where the weighting question of plans/o63-gauge-weight-audit.md lives,
and every variant below is post-processing on the same S:
  default        W = kept mask (0/1), iid                       -- the o63 weighting
  --sigma-obs S  every eigenvalue scales by (0.15/S)^2: the count changes,
                 the eigenvectors do not
  --ar1 RHO      W = blockdiag over gauges of C^-1, C an AR(1) correlation
                 with lag-1 coefficient RHO over the gauge's kept times: the
                 independence correction, which changes vectors as well
  --demean       W = P, the projector that removes each gauge's mean residual:
                 a per-gauge offset (datum / representation error) is a
                 nuisance parameter and what remains is hydrograph shape
  --sigma-scan   the count above unity as a function of sigma, so the table
                 the paper gives for the marks (tab:scaling) exists for the
                 gauges too

With --marks, the mark sensitivity matrix is assembled from the o58/o61 peak
dumps at the SAME prior and the two spectra are compared: the angle between
the leading eigenvectors, the overlap of each observable's informative
subspace (lambda > 1) with the other's, and the sign of the two objectives'
gradients along the gauge leading direction. That is Emil's question --
complementary, redundant, or opposed -- as numbers.
"""
import sys, re
import numpy as np

NLCD = {11: .038, 21: .040, 22: .090, 23: .120, 24: .160, 31: .027, 41: .150,
        42: .120, 43: .140, 52: .115, 71: .038, 81: .038, 82: .035, 90: .098, 95: .068}


def flag(name, default=None, nargs=1):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if nargs == 0:
        return True
    return sys.argv[i + 1]


def read_table(path):
    with open(path) as f:
        ng, K = map(int, f.readline().split())
        cells = list(map(int, f.readline().split()))
        rows = [f.readline().split() for _ in range(K)]
    t = np.array([float(r[0]) for r in rows])
    W = np.array([[float(x) for x in r[1:]] for r in rows])   # K x ng
    assert W.shape == (K, ng), path
    return cells, t, W


def read_zb(path):
    zb = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        g, cell, z = line.split()
        zb[int(g)] = float(z)
    return np.array([zb[g] for g in range(len(zb))])


def read_peaks(path):
    mark, w, obs, mod, step = [], [], [], [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        mark.append(int(f[0])); w.append(float(f[1])); obs.append(float(f[2]))
        mod.append(float(f[3])); step.append(float(f[4]))
    o = np.argsort(mark)
    return np.array(mark)[o], np.array(w)[o], np.array(obs)[o], np.array(mod)[o], np.array(step)[o]


def code_of(path):
    m = re.search(r'col(\d+)', path)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- arguments
consumed = set()
for name, n in (('--sigma-obs', 1), ('--sigma-alpha', 1), ('--ar1', 1), ('--demean', 0), ('--no-mask', 0),
                ('--sigma-scan', 0), ('--marks', 0), ('--drop', 1), ('--central', 0)):
    if name in sys.argv:
        i = sys.argv.index(name); consumed.add(i)
        if n: consumed.add(i + 1)
pos = [a for i, a in enumerate(sys.argv[1:], 1) if i not in consumed]
# --marks takes everything after it that is a peak dump; split positionals there
if '--marks' in sys.argv:
    im = sys.argv.index('--marks')
    gauge_pos = [a for i, a in enumerate(sys.argv[1:], 1) if i not in consumed and i < im]
    mark_pos = [a for i, a in enumerate(sys.argv[1:], 1) if i not in consumed and i > im]
else:
    gauge_pos, mark_pos = pos, []

SIGMA = flag('--sigma-obs'); SIG_A = flag('--sigma-alpha')
if SIGMA is None or SIG_A is None:
    sys.exit('ERROR: --sigma-obs and --sigma-alpha are required (the runs used 0.15 m and 0.30).')
SIGMA = float(SIGMA); SIG_A = float(SIG_A)
RHO = flag('--ar1'); RHO = float(RHO) if RHO is not None else None
DEMEAN = flag('--demean', False, 0); NOMASK = flag('--no-mask', False, 0); SCAN = flag('--sigma-scan', False, 0)
# --drop 24,95   : leave these classes out of S (o65: the +5% class-24 column is 12x its
#                  adjoint prediction -- a nonlinear response, not a sensitivity)
# --central      : where a matching -eps column exists (o66 names them col<C>_e-<eps>.txt),
#                  use the central difference (f(+eps) - f(-eps)) / 2 eps for that class
DROP = {int(c) for c in flag('--drop', '').split(',') if c}
CENTRAL = flag('--central', False, 0)

eps = float(gauge_pos[0]); obs_path = gauge_pos[1]; base_path = gauge_pos[2]; col_paths = gauge_pos[3:]

# ---------------------------------------------------------------- gauge S
cells_o, t_o, OBS = read_table(obs_path)
cells_b, t_b, MOD0 = read_table(base_path)
zb = read_zb(base_path + '.zb')
assert cells_o == cells_b and np.allclose(t_o, t_b), 'base dump does not match the observation table'
K, ng = OBS.shape
mask = ~np.isnan(OBS)
if not NOMASK:
    mask &= (OBS - zb[None, :]) > 0.0           # -adjoint_obs_above_bed, min depth 0
kept = int(mask.sum())
print(f'{ng} gauges x {K} times = {ng*K} (gauge,time) pairs; {kept} kept'
      f' ({"NaN only" if NOMASK else "NaN + above-bed"} mask)')

# self-check: masked J of the base dump against the driver's printed J
r0 = np.where(mask, MOD0 - OBS, 0.0)
J0 = 0.5 * (r0**2).sum() / SIGMA**2
print(f'base: J {J0:.6e}, RMSE {SIGMA*np.sqrt(2*J0/kept):.4f} m  <- must equal the driver\'s "gauge eval" line'
      f' (o64 prior: 3.131332e+04, 3.243 m)')
print('per-gauge residual of the base field (kept records only):')
for g in range(ng):
    m = mask[:, g]
    if m.sum():
        rg = (MOD0[m, g] - OBS[m, g])
        print(f'  gauge {g:2d} cell {cells_b[g]:8d}: {int(m.sum()):2d} kept, mean {rg.mean():+7.3f} m, rms {np.sqrt((rg**2).mean()):6.3f} m,'
              f' obs depth {(OBS[m,g]-zb[g]).min():.2f}-{(OBS[m,g]-zb[g]).max():.2f} m')

cols, minus = {}, {}
for p in col_paths:
    c = code_of(p)
    if c is None:
        print(f'  skipping {p}: no col<CODE> in the name'); continue
    me = re.search(r'_e(-?[0-9.]+)\.txt$', p)     # signed-eps columns from o66
    if me and abs(float(me.group(1))) != eps:
        print(f'  skipping {p}: eps {me.group(1)} is not +/-{eps}'); continue
    cc, tt, M = read_table(p)
    assert cc == cells_b and np.allclose(tt, t_b), p
    if me and float(me.group(1)) < 0:
        minus[c] = M
    else:
        cols[c] = M
for c in DROP:
    if c in cols:
        cols.pop(c); print(f'  dropping class {c} from S (--drop)')
codes = sorted(cols)
Kc = len(codes)
S_full = np.zeros((K * ng, Kc))                  # row index i = k*ng + g
central_used = []
for j, c in enumerate(codes):
    if CENTRAL and c in minus:
        S_full[:, j] = ((cols[c] - minus[c]) / (2 * eps)).reshape(-1); central_used.append(c)
    else:
        S_full[:, j] = ((cols[c] - MOD0) / eps).reshape(-1)
if central_used:
    print(f'central differences used for classes {central_used}; one-sided +{eps} for the rest')
    for c in central_used:   # the nonlinearity check the o66 columns exist for
        fp = ((cols[c] - MOD0) / eps).reshape(-1)
        fm = ((MOD0 - minus[c]) / eps).reshape(-1)
        mm = mask.reshape(-1)
        print(f'  class {c}: |S| kept rows from +eps {np.linalg.norm(fp[mm]):.3f}, from -eps {np.linalg.norm(fm[mm]):.3f}'
              f' (equal if linear); J(+eps)-J0 vs J0-J(-eps) is in the slurm logs')
mvec = mask.reshape(-1)
print(f'{Kc} classes; |S| per class over kept rows (m per unit alpha): '
      + ', '.join(f'{c}:{np.linalg.norm(S_full[mvec, j]):.2f}' for j, c in enumerate(codes)))


def weight_matrix(mask, rho=None, demean=False):
    """W on the kept rows only (n_kept x n_kept), in the row order of mvec."""
    idx = np.where(mask.reshape(-1))[0]
    n = len(idx)
    W = np.eye(n)
    gauge_of = idx % ng
    if rho is not None:
        W = np.zeros((n, n))
        for g in range(ng):
            sel = np.where(gauge_of == g)[0]
            if not len(sel):
                continue
            m = len(sel)
            C = rho ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
            W[np.ix_(sel, sel)] = np.linalg.inv(C)
    if demean:
        P = np.eye(n)
        for g in range(ng):
            sel = np.where(gauge_of == g)[0]
            if len(sel):
                P[np.ix_(sel, sel)] -= 1.0 / len(sel)
        W = P @ W @ P
    return W, idx


def spectrum(S, W, sigma, sig_alpha):
    H = S.T @ W @ S / sigma**2
    Hw = (sig_alpha**2) * H                       # uniform relative prior: Gamma = sig_alpha^2 I
    lam, V = np.linalg.eigh(Hw)
    o = np.argsort(lam)[::-1]
    return lam[o], V[:, o], H


def report(lam, V, codes, title):
    print(f'\n{title}')
    print(f"{'i':>3} {'lambda':>10} {'data vs prior':<17} {'shrink':>7}  leading classes")
    for i in range(len(lam)):
        L = max(lam[i], 0.0)
        tag = ('data-determined' if L > 10 else 'data > prior' if L > 1 else
               'comparable' if L > 0.1 else 'prior-determined')
        top = np.argsort(np.abs(V[:, i]))[::-1][:4]
        lead = ', '.join(f'{codes[t]}({V[t,i]:+.2f})' for t in top)
        print(f'{i:>3} {lam[i]:>10.4g} {tag:<17} {1/np.sqrt(1+L):>6.2f}x  {lead}')
    n1 = int((lam > 1).sum()); dofs = float((np.maximum(lam, 0) / (1 + np.maximum(lam, 0))).sum())
    print(f'  eigenvalues > 1 : {n1} of {len(lam)}   <- roughness combinations this observable determines')
    print(f'  degrees of freedom for signal (Rodgers): {dofs:.2f}')
    if len(lam) > 1 and lam[1] > 0:
        print(f'  spectral gap lambda_0/lambda_1 = {lam[0]/lam[1]:.2f}')
    return n1


W, idx = weight_matrix(mask, RHO, DEMEAN)
S = S_full[idx, :]
lam_g, V_g, H_g = spectrum(S, W, SIGMA, SIG_A)
wdesc = f'sigma_obs {SIGMA} m, sigma_alpha {SIG_A}' + (f', AR(1) rho {RHO}' if RHO is not None else ', iid') + (', per-gauge demeaned' if DEMEAN else '')
n1_g = report(lam_g, V_g, codes, f'GAUGE prior-preconditioned Gauss-Newton spectrum ({wdesc}):')

if SCAN:
    print('\nsigma scan (count of lambda > 1; eigenvectors fixed):')
    for s in (0.15, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0):
        lam_s = lam_g * (SIGMA / s)**2
        print(f'  sigma {s:4.2f} m: {int((lam_s > 1).sum()):2d} supported, lambda_0 {lam_s[0]:.3g}, dofs {float((lam_s/(1+lam_s)).sum()):.2f}')

# Gauss-Newton prediction of where a gauge calibration of the three o63 classes stops,
# for any weighting: needs the gradient at the prior, g = S^T W r0 / sigma^2 (same S).
r_kept = r0.reshape(-1)[idx]
g_alpha = S.T @ W @ r_kept / SIGMA**2
print('\ngauge gradient at the prior from S and the base residual, dJ/dalpha per class '
      '(check vs o64: 22 -4016, 23 -6376, 90 -676 at sigma 0.15 iid):')
print('  ' + ', '.join(f'{c}:{g_alpha[j]:+.0f}' for j, c in enumerate(codes)))
act = [codes.index(c) for c in (22, 23, 90) if c in codes]
if len(act) == 3:
    Ha = H_g[np.ix_(act, act)] + np.eye(3) / SIG_A**2
    da = -np.linalg.solve(Ha, g_alpha[act])
    print(f'  GN step for classes 22/23/90 with the prior, this weighting: alpha = '
          + ', '.join(f'{1+d:.2f}' for d in da) + '   (o63 reached 3.00, 3.00, 1.74 on the box)')

# ---------------------------------------------------------------- marks, and the comparison
if mark_pos:
    mark, wm, obsm, mod0, step0 = read_peaks(mark_pos[0])
    Sm = np.zeros((len(mark), Kc))
    for p in mark_pos[1:]:
        c = code_of(p)
        if c not in cols:
            continue
        _, _, _, modk, _ = read_peaks(p)
        Sm[:, codes.index(c)] = (modk - mod0) / eps
    lam_m, V_m, H_m = spectrum(Sm, np.diag(wm), SIGMA, SIG_A)
    n1_m = report(lam_m, V_m, codes, f'MARK prior-preconditioned Gauss-Newton spectrum (sigma_obs {SIGMA} m, sigma_alpha {SIG_A}), same S construction:')

    print('\nCOMPARISON of the two observables (Emil: complementary, redundant, or opposed?)')
    cosang = abs(V_g[:, 0] @ V_m[:, 0])
    print(f'  angle between leading eigenvectors: {np.degrees(np.arccos(min(1.0, cosang))):.1f} deg (|cos| = {cosang:.3f})')
    kg = max(n1_g, 1); km = max(n1_m, 1)
    Ug = V_g[:, :kg]; Um = V_m[:, :km]
    print(f'  informative subspaces: gauges {kg}-dim, marks {km}-dim (lambda > 1, at least the leading vector)')
    print(f'  fraction of the mark subspace inside the gauge subspace: {np.linalg.norm(Ug.T @ Um)**2 / km:.3f}')
    print(f'  fraction of the gauge subspace inside the mark subspace: {np.linalg.norm(Um.T @ Ug)**2 / kg:.3f}')
    # sign: the two misfit gradients along the gauge leading direction, at the prior
    rm = np.where(wm > 0, mod0 - obsm, 0.0)
    g_m = Sm.T @ np.diag(wm) @ rm / SIGMA**2
    v = V_g[:, 0]
    print(f'  along the gauge leading direction v_0: dJ_gauge/dv = {g_alpha @ v:+.0f}, dJ_marks/dv = {g_m @ v:+.0f}'
          f'  -> {"OPPOSED" if (g_alpha @ v) * (g_m @ v) < 0 else "same sign"}')
    v = V_m[:, 0]
    print(f'  along the mark  leading direction u_0: dJ_gauge/du = {g_alpha @ v:+.0f}, dJ_marks/du = {g_m @ v:+.0f}'
          f'  -> {"OPPOSED" if (g_alpha @ v) * (g_m @ v) < 0 else "same sign"}')
    print('  mark gradient per class from S (check vs o62: 22 -29.5, 23 +90.7, 90 +66.2 in alpha): '
          + ', '.join(f'{c}:{g_m[j]:+.1f}' for j, c in enumerate(codes)))
    print(f'  joint spectrum (both observables together): '
          + ', '.join(f'{l:.3g}' for l in np.sort(np.linalg.eigvalsh(SIG_A**2 * (H_g + H_m)))[::-1][:5]) + ' ...')
