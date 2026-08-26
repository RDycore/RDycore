#!/usr/bin/env python3
"""Assemble the class Hessian from o56's gradient dumps and report its spectrum.

  o56_hessian.py <eps> <base dump> <column dumps...>

Each dump is "<NLCD code> <manning n> <dJ/dn>" as written by
-adjoint_classes_grad_dump. Columns must be named o56_g_col<CODE>.txt so the
perturbed class can be identified.

WHAT THIS COMPUTES, and why the preconditioning matters.

The optimizer's variable is alpha_k = n_k/n_prior_k, so first convert the
dumped dJ/dn to dJ/dalpha = n_prior * dJ/dn. Differencing those gives the
Hessian in alpha,

    H[j,k] ~ ( g_j(alpha + eps e_k) - g_j(alpha) ) / eps .

H is then symmetrized -- one-sided differences do not produce a symmetric
matrix, and the asymmetry is a useful error estimate, so it is reported.

The eigenvalues of H alone are not interpretable: they carry the units of the
objective and say nothing about whether the data beat the prior. The
meaningful object is the PRIOR-PRECONDITIONED misfit Hessian. With a Gaussian
prior of covariance Gamma on the parameters, the posterior precision is
H_mis + Gamma^-1, and in the whitened basis this is I + Gamma^(1/2) H_mis
Gamma^(1/2). Each eigenvalue lambda_i of that whitened misfit Hessian
compares data information against prior information in direction i:

    lambda_i >> 1  : the data determines this direction
    lambda_i ~  1  : data and prior contribute about equally
    lambda_i << 1  : the prior determines it; the data says nothing

so the count of lambda_i > 1 is the number of parameters the observable
actually supports -- the quantity this project has so far inferred from scans.
The eigenvectors say which COMBINATIONS of land-cover classes those are,
which no scan can. Rodgers calls the trace of lambda/(1+lambda) the degrees
of freedom for signal; it is reported as the non-integer version of the count.

The relative-error reduction per direction, 1/sqrt(1+lambda), is the factor by
which calibration shrinks the prior uncertainty in that direction -- i.e. the
error bars the calibration earns.

VALIDATION, which comes free. The dumped gradient includes the Tikhonov term,
so what is assembled is the POSTERIOR Hessian and the whitened prior is
exactly the identity. A direction the data does not constrain must therefore
return lambda_total = 1 to within finite-difference error. On the Houston twin
the five weakest directions -- single-class directions of the classes with
negligible gradient -- return 1.0011, 1.0015, 1.0042, 1.0049, 1.0096. Getting
the chain rule, the assembly or the whitening wrong would move those off 1.
"""
import sys, re
import numpy as np


def read_dump(path):
    codes, n, g = [], [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        codes.append(int(f[0])); n.append(float(f[1])); g.append(float(f[2]))
    return np.array(codes), np.array(n), np.array(g)


eps = float(sys.argv[1])
base_path, col_paths = sys.argv[2], sys.argv[3:]
codes, n_prior, g0_n = read_dump(base_path)
K = len(codes)
idx = {c: i for i, c in enumerate(codes)}

# dJ/dn -> dJ/dalpha
g0 = g0_n * n_prior
H = np.full((K, K), np.nan)
for p in col_paths:
    m = re.search(r"col(\d+)", p)
    if not m:
        print(f"  skipping {p}: no col<CODE> in the name"); continue
    c = int(m.group(1))
    if c not in idx:
        print(f"  skipping {p}: class {c} not in the base dump"); continue
    ck, _, gk_n = read_dump(p)
    assert list(ck) == list(codes), f"{p}: class ordering differs from base"
    H[:, idx[c]] = ((gk_n * n_prior) - g0) / eps

missing = [codes[k] for k in range(K) if np.isnan(H[:, k]).any()]
if missing:
    print(f"WARNING: no column for classes {missing}; dropping them")
    keep = [k for k in range(K) if not np.isnan(H[:, k]).any()]
    H = H[np.ix_(keep, keep)]; codes = codes[keep]; n_prior = n_prior[keep]; K = len(keep)

asym = np.abs(H - H.T).max() / np.abs(H).max()
Hs = 0.5 * (H + H.T)
print(f"assembled {K}x{K} Hessian in alpha; relative asymmetry {asym:.2e} "
      f"({'fine -- one-sided FD' if asym < 0.2 else 'LARGE: suspect eps or solver noise'})")

# prior covariance in alpha. sigma_n = 0.015 absolute -> per-class sigma_alpha
SIGMA_N = 0.015
sig_alpha = SIGMA_N / n_prior
Hw = (sig_alpha[:, None] * Hs) * sig_alpha[None, :]     # Gamma^1/2 H Gamma^1/2
lam_tot, V = np.linalg.eigh(Hw)
order = np.argsort(lam_tot)[::-1]; lam_tot, V = lam_tot[order], V[:, order]

# The dumped gradient includes the Tikhonov term, so Hw is the whitened
# POSTERIOR Hessian, not the misfit one. In whitened coordinates the prior
# contributes exactly the identity -- for the absolute prior, sigma_alpha_k =
# sigma_n/n_prior_k and the prior Hessian in alpha is beta*n_prior_k^2, whose
# whitening gives sigma_n^2 * beta = 1 on the diagonal. So the misfit spectrum
# is lam_tot - 1, and an unconstrained direction must come out at exactly 1.
# That is a free end-to-end check on the Hessian assembly, the alpha chain
# rule and the whitening: the smallest eigenvalues landing on 1.000 means all
# three are right.
lam = lam_tot - 1.0
print(f"  self-check: smallest whitened eigenvalue {lam_tot[-1]:.6f} "
      f"(must be 1 for a direction the data does not constrain; "
      f"{'OK' if abs(lam_tot[-1]-1) < 0.05 else 'SUSPECT'})")

print(f"\nprior-preconditioned MISFIT-Hessian spectrum (sigma_n = {SIGMA_N}):")
print(f"{'i':>3} {'lambda':>12} {'data vs prior':<16} {'err reduction':>13}  leading classes")
for i in range(K):
    L = lam[i]
    tag = ("data-determined" if L > 10 else "data > prior" if L > 1 else
           "comparable" if L > 0.1 else "prior-determined")
    red = 1.0 / np.sqrt(1.0 + max(L, 0.0))   # posterior/prior std in this direction
    top = np.argsort(np.abs(V[:, i]))[::-1][:3]
    lead = ", ".join(f"{codes[t]}({V[t,i]:+.2f})" for t in top)
    print(f"{i:>3} {L:>12.4g} {tag:<16} {red:>12.2f}x  {lead}")

n_gt1 = int((lam > 1).sum())
dofs = float((lam / (1.0 + lam)).sum())
print(f"\n  eigenvalues > 1 : {n_gt1} of {K}   <- parameters the observable supports")
print(f"  degrees of freedom for signal (Rodgers): {dofs:.2f}")
if K > 1 and lam[1] > 0: print(f"  spectral decay lambda_0/lambda_1 = {lam[0]/lam[1]:.1f}")
print(f"\n  Compare with the scan-based claim: the uniform-alpha and half-domain")
print(f"  experiments were read as ~ONE supported degree of freedom.")
