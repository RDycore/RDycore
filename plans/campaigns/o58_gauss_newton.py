#!/usr/bin/env python3
"""Gauss-Newton Hessian and its prior-preconditioned spectrum, from peak dumps.

  o58_gauss_newton.py <eps> <base peak dump> <column dumps...> [--sigma-obs S] [--sigma-n S]

sigma-obs MUST match the -adjoint_obs_error the runs used (production: 0.15;
the driver's own default if the flag is omitted is 0.01). Getting it wrong
rescales every eigenvalue by (ratio)^2 -- a factor of 225 between those two --
which silently changes the supported-parameter count, so it is required to be
passed explicitly rather than assumed.

Each dump is "<mark> <weight> <h_obs> <h_model_peak> <peak_step>", as written
by -adjoint_hwm_dump. Columns must be named o58_pk_col<CODE>.txt.

WHAT THIS COMPUTES

The observation-sensitivity matrix over the marks and the land-cover classes,

    S[i,k] = ( peak_i(alpha + eps e_k) - peak_i(alpha) ) / eps ,

and from it the Gauss-Newton Hessian of the misfit, H = S^T W S / sigma^2,
with W the 0/1 mark weight. H is positive semi-definite by construction, so
unlike a full Hessian differenced from the gradient field it cannot return
negative eigenvalues -- which is the failure mode that invalidated o56.

Whitening by the prior covariance Gamma gives Gamma^(1/2) H Gamma^(1/2), whose
eigenvalues compare data information against prior information direction by
direction:

    lambda >> 1 : data determines it        lambda << 1 : the prior does

so the count above 1 is the number of parameters the observable supports, and
the eigenvectors say which COMBINATIONS of classes those are. The trace of
lambda/(1+lambda) is Rodgers' degrees of freedom for signal; 1/sqrt(1+lambda)
is the factor by which calibration shrinks the prior uncertainty in that
direction, i.e. the error bars the calibration earns.

THE ARGMAX CHECK, which is the reason this is trustworthy where o56 was not.
The peak misfit is differentiable only where each mark's peak TIME is locally
constant. The dumps carry peak_step, so this reports how many marks moved
their peak in each column. A column with many moved peaks is sampling the
nondifferentiability rather than the derivative, and its sensitivities should
be distrusted -- so the number is printed rather than assumed to be zero.
"""
import sys, re
import numpy as np

def _argval(flag, default):
    return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


SIGMA = _argval("--sigma-obs", None) if "--sigma-obs" in sys.argv else None
SIGMA_N = _argval("--sigma-n", 0.015)
if SIGMA is None:
    sys.exit("ERROR: pass --sigma-obs explicitly (the -adjoint_obs_error the runs used;\n"
             "       production is 0.15, the driver default is 0.01). Every eigenvalue\n"
             "       scales as 1/sigma^2, so guessing it changes the answer.")


def read_peaks(path):
    mark, w, obs, mod, step = [], [], [], [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        mark.append(int(f[0])); w.append(float(f[1])); obs.append(float(f[2]))
        mod.append(float(f[3])); step.append(float(f[4]))
    o = np.argsort(mark)
    return (np.array(mark)[o], np.array(w)[o], np.array(obs)[o],
            np.array(mod)[o], np.array(step)[o])


argv = [a for a in sys.argv[1:] if not a.startswith("--")]
skip = set()
for f in ("--sigma-obs", "--sigma-n"):
    if f in sys.argv:
        skip.add(sys.argv[sys.argv.index(f) + 1])
argv = [a for a in argv if a not in skip]
eps = float(argv[0])
mark, w, obs, mod0, step0 = read_peaks(argv[1])
N = len(mark)

cols, codes = {}, []
for p in argv[2:]:
    m = re.search(r"col(\d+)", p)
    if not m:
        print(f"  skipping {p}: no col<CODE> in the name"); continue
    c = int(m.group(1))
    mk, _, _, modk, stepk = read_peaks(p)
    assert list(mk) == list(mark), f"{p}: mark ordering differs from base"
    cols[c] = (modk, stepk); codes.append(c)
codes.sort()
K = len(codes)

S = np.zeros((N, K))
moved = {}
for j, c in enumerate(codes):
    modk, stepk = cols[c]
    S[:, j] = (modk - mod0) / eps
    moved[c] = int(((stepk != step0) & (w > 0)).sum())

nw = int((w > 0).sum())
print(f"{N} marks ({nw} weighted), {K} classes, eps = {eps}")
print(f"argmax check -- marks whose peak TIME moved, per column (of {nw}):")
bad = [f"{c}:{moved[c]}" for c in codes if moved[c] > 0]
print("  " + (", ".join(bad) if bad else "none -- every mark kept its peak time"))
tot_moved = sum(moved.values())
print(f"  total {tot_moved} of {nw*K} mark-column pairs "
      f"({100*tot_moved/(nw*K):.1f}%); the peak misfit is differentiable where this is 0")

# Gauss-Newton Hessian of the misfit, in alpha
W = np.diag(w)
H = (S.T @ W @ S) / SIGMA**2
ev_raw = np.linalg.eigvalsh(H)
print(f"\nGauss-Newton Hessian: symmetric by construction; "
      f"min eigenvalue {ev_raw.min():.4g} "
      f"({'PSD as required' if ev_raw.min() > -1e-8 * max(ev_raw.max(),1) else 'NEGATIVE -- impossible, check inputs'})")

# prior. sigma_n absolute -> per-class sigma_alpha = sigma_n / n_prior.
# n_prior is read from the class table used for the base run if available;
# fall back to the published NLCD lookup.
NLCD = {11: .038, 21: .040, 22: .090, 23: .120, 24: .160, 31: .027, 41: .150,
        42: .120, 43: .140, 52: .115, 71: .038, 81: .038, 82: .035, 90: .098, 95: .068}
n_prior = np.array([NLCD[c] for c in codes])
sig_alpha = SIGMA_N / n_prior

Hw = (sig_alpha[:, None] * H) * sig_alpha[None, :]
lam, V = np.linalg.eigh(Hw)
o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]

print(f"\nprior-preconditioned Gauss-Newton spectrum (sigma_n = {SIGMA_N}, sigma_obs = {SIGMA}):")
print(f"{'i':>3} {'lambda':>12} {'data vs prior':<17} {'err reduction':>13}  leading classes")
for i in range(K):
    L = max(lam[i], 0.0)
    tag = ("data-determined" if L > 10 else "data > prior" if L > 1 else
           "comparable" if L > 0.1 else "prior-determined")
    top = np.argsort(np.abs(V[:, i]))[::-1][:4]
    lead = ", ".join(f"{codes[t]}({V[t,i]:+.2f})" for t in top)
    print(f"{i:>3} {lam[i]:>12.4g} {tag:<17} {1/np.sqrt(1+L):>12.2f}x  {lead}")

n1 = int((lam > 1).sum())
dofs = float((np.maximum(lam, 0) / (1 + np.maximum(lam, 0))).sum())
print(f"\n  eigenvalues > 1 : {n1} of {K}   <- parameters the observable supports")
print(f"  degrees of freedom for signal (Rodgers): {dofs:.2f}")
if K > 1 and lam[1] > 0:
    print(f"  spectral gap lambda_0/lambda_1 = {lam[0]/lam[1]:.2f}")

# is the leading direction the uniform mode the alpha-scan assumed?
u = np.ones(K) / np.sqrt(K)
print(f"\n  |<v_0, uniform>| = {abs(V[:,0] @ u):.3f}  "
      f"(1.0 would mean the leading mode IS the uniform roughness scale the")
print(f"  alpha scan measured; near 0 means the scan probed a different direction)")
