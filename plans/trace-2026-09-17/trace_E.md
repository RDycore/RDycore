# Number provenance audit — manning-calibration.tex lines 1647–2146 (Sec 6.4 spectrum, Sec 6.5 cross-observable)

Source shorthand:
- `SPEC-N` = `/Users/markadams/Codes/RDycore-gpu/logs/o61/o61_spectrum_sigma_n0.015.txt`
- `SPEC-A30` = `/Users/markadams/Codes/RDycore-gpu/logs/o61/o61_spectrum_sigma_alpha0.30.txt` (same line layout as SPEC-N)
- `SPEC-A10/15/20/50` = the corresponding `o61_spectrum_sigma_alpha0.{10,15,20,50}.txt`
- `RES` = `/Users/markadams/Codes/RDycore-gpu/plans/RESULTS-gpu-implicit.md`
- `AUDIT` = `/Users/markadams/Codes/RDycore-gpu/plans/o63-gauge-weight-audit.md`
- `HANDOFF` = `/Users/markadams/Codes/RDycore-gpu/plans/PAPER-SESSION-HANDOFF.md`
- `PK` = the sixteen peak dumps `/Users/markadams/Codes/RDycore-gpu/logs/o61/o58_e0.05w43200_pk_{base,col*}.txt`
- Recomputation script written for this audit: `/private/tmp/claude-501/-Users-markadams-Codes-RDycore-gpu/e56cd854-8c6f-4898-9d97-81630bef32a2/scratchpad/chk.py`

---

## ¶ Opening of Sec 6.4 (lines 1647–1672)

| line | number as printed | context | source (file:line, value quoted) | status |
|---|---|---|---|---|
| 1662 | $\lambda_i > 1$ | threshold for a supported parameter | SPEC-N:26 `eigenvalues > 1 : 1 of 15   <- parameters the observable supports` | EXACT (definition used by the analysis script) |
| 1668 | $1$ to $3$ | α traversed by the calibrations | RES:2465–2468 (`22 -> 0.27 = 3x`), tab:threefifteen box `[0.3, 3]`; RES:2482 `1.63x -> 2.34x -> 3x` | EXACT |
| 1670 | $35\%$ | step moved 35% of linear prediction | AUDIT:66 `first-order predicted decrease -21283, measured -7354: 35% of linear`; HANDOFF:30 `(o63 step: 35% of linear)` | CONTRADICTED IN CONTEXT — the 35% is the **gauge** objective along the gauge step (−7354/−21283 = 34.6%). The sentence cross-refs Sec 6.5, whose own printed pair is +171 predicted / +87 measured = **51%**, not 35%. Number is sourced; the pointer is wrong. |

## ¶ "Assembling it" (lines 1674–1709)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1681 | $45\%$ | asymmetry of the o56 gradient-differenced Hessian | RES:1971 `returned 45% relative asymmetry` | EXACT |
| 1681 | $-6.8$ | most negative eigenvalue of that matrix | RES:1972 `eigenvalues to -6.8` | EXACT |
| 1694 | $5\%$ | per-class perturbation for the columns | SPEC-N:1 `eps = 0.05`; RES:2610 `each class raised 5%, as o58/o61` | EXACT |
| 1695 | 16 forwards | cost of the GN assembly | RES:1983 `costs 16 FORWARDS rather than 16 forward+adjoints`; PK = 1 base + 15 columns | EXACT |
| 1700 | $4\%$ | S-gradient vs adjoint agreement | RES:2675 `reproduces the o62 adjoint gradient on every class to within 4%`; RES:2820 same | EXACT |
| 1701 | $-28.4$ / $-29.5$ | developed-low, S vs adjoint | RES:2676 / RES:2821 `22 -28.4 vs -29.5` | EXACT |
| 1701 | $+89.7$ / $+90.7$ | developed-medium | RES:2676 / RES:2821 `23 +89.7 vs +90.7` | EXACT |
| 1702 | $+65.1$ / $+66.2$ | woody wetland | RES:2676 / RES:2821 `90 +65.1 vs +66.2` | EXACT |
| 1704–05 | $5\%$ (twice) | one-sided secants | SPEC-N:1 `eps = 0.05` | EXACT |
| 1708 | (gauge construction failed the test) | no gauge spectrum in the paper | RES:2778–2783 `RETRACTED from o65 ... A valid gauge Gauss-Newton spectrum has not been computed` | EXACT (qualitative, sourced) |

## ¶ The argmax check (lines 1711–1734)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1714 | $156$ of $690$ | mark–column pairs moving peak time | SPEC-N:4 `total 156 of 690 mark-column pairs (22.6%)`; recomputed from PK: `moved 156 / 690` | EXACT |
| 1714 | $22.6\%$ | same | SPEC-N:4 `(22.6%)` | EXACT |
| 1715 | $27$ | pilot argmax moves | RES:1984–85 `27 of 690 mark-column pairs (3.9%) changed their peak time` | EXACT |
| 1715 | $3.9\%$ | same | RES:1985 `(3.9%)` | EXACT |
| 1716 | two-event-hour | pilot window | RES:2374–76 `over 7,200 steps (2 event-hours)` | EXACT |
| 1719 | twelve | production window hours | RES:2015 `12-hr cluster-A window`; 43,200 steps at dt = 1 s | EXACT |
| 1722 | $156$ | restated | SPEC-N:4 | EXACT |
| 1722 | $107$ | pairs shifting one sample | RES:2387 `shift 300 steps (1 obs sample): 107 pairs (71%)` | **CONTRADICTED** — 107 belongs to the earlier **14-column, 151-of-644** analysis (RES:2383 `151 of 644 ... (23.4%)`). Recomputed on the full 15-column 156-of-690 set (chk.py): **112** pairs shift by 300 steps. |
| 1722 | $71\%$ | share of the 156 | RES:2387 `(71%)` (of 151); recomputed on 690: 112/156 = **71.8%** | ROUNDED/OK — the percentage survives the change of sample (71.8% → 71%), only the count 107 does not (107/156 = 68.6%). |
| 1723 | $300$-step | one observation sample | RES:2387 `shift 300 steps (1 obs sample)`; 144 obs times over 43,200 steps | EXACT |
| 1724 | $2.3$ mm | median change in peak value | RES:2390 `median \|change in peak value\|: 2.3e-3 m`; recomputed from PK: 0.002253 m | EXACT |
| 1724 | $1.5$ m | typical peak | RES:2391 `typical peak magnitude: 1.5 m`; recomputed: median base peak over the moved pairs = 1.50 m | EXACT |
| 1725 | two parts in a thousand | 2.3 mm / 1.5 m | DERIVED: 0.0023/1.5 = 0.15% = 1.5 parts per thousand (RES:2394 calls it `0.15%`) | DERIVED (rounded up from 1.5 to "two") |
| 1726 | ten pairs | multi-hour relocations | RES:2389 `shift 2100-6900 steps: 10 pairs`; recomputed on 690: shifts ≥ 2100 steps = **10** | EXACT |
| 1726 | $1.6\%$ | share of those ten | RES:2404 `are 1.6% of the sample` — but that sample was 644 (10/644 = 1.55%) | **CONTRADICTED** — against the paper's own denominator of 690, 10/690 = **1.4%**. |
| 1726 | more than an hour | size of those relocations | RES:2389 bin is `2100-6900 steps`; at dt = 1 s an hour = 3600 steps. Recomputed: only **7** of the 10 exceed 3600 steps (two at 2100 = 35 min, one at 3300 = 55 min) | **CONTRADICTED** — "ten pairs relocate by more than an hour" is true of seven; the other three shift 35–55 min. |

## ¶ fig:spectrum caption (lines 1740–1751) and `fig_spectrum.tex`

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1742 | $\sigma_n = 0.015$ | filled series prior | SPEC-N:8 `sigma_n = 0.015 absolute, sigma_obs = 0.15` | EXACT |
| 1743 | $\pm 30\%$ | open series prior | SPEC-A30:8 `sigma_alpha = 0.3 uniform` | EXACT |
| 1745 | $\lambda = 1$ | threshold line | SPEC-N:26 definition | EXACT |
| 1746 | One direction … three | counts at the two priors | SPEC-N:26 `1 of 15`; SPEC-A30:26 `3 of 15` | EXACT |
| 1747 | three decades | fall-off below the gap | DERIVED: 0.6823 → 0.0003867 is 3.25 decades (SPEC-N:11, 24) | DERIVED |
| fig_spectrum.tex:46–49, 55–58 | 2.779, 0.6823, 0.6127, 0.324, 0.2255, 0.09968, 0.06136, 0.03434, 0.02256, 0.01849, 0.01133, 0.007279, 0.003104, 0.002383, 0.0003867 | filled series | SPEC-N:10–24, verbatim all fifteen | EXACT (all 15 match to every digit) |
| fig_spectrum.tex:64–67 | 12.73, 2.985, 1.128, 0.8529, 0.2548, 0.2386, 0.1577, 0.0876, 0.05021, 0.02444, 0.01454, 0.01121, 0.006023, 0.004086, 0.0005972 | open series | SPEC-A30:10–24, verbatim all fifteen | EXACT (all 15 match to every digit) |

## ¶ "The count depends on the prior width" (lines 1754–1772)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1755 | $\sigma_n = 0.015$ | absolute prior | SPEC-N:8 | EXACT |
| 1756 | $2.78$ | λ₀ | SPEC-N:10 `2.779` | ROUNDED (2.779) |
| 1756 | $0.68$ | λ₁ | SPEC-N:11 `0.6823` | ROUNDED |
| 1756 | $0.61$ | λ₂ | SPEC-N:12 `0.6127` | ROUNDED |
| 1756 | $0.32$ | λ₃ | SPEC-N:13 `0.324` | ROUNDED |
| 1756 | $0.23$ | λ₄ | SPEC-N:14 `0.2255` | ROUNDED |
| 1756 | $0.10$ | λ₅ | SPEC-N:15 `0.09968` | ROUNDED |
| 1757 | a single eigenvalue above unity | count | SPEC-N:26 `1 of 15` | EXACT |
| 1758 | $4.1$ | spectral gap | SPEC-N:28 `spectral gap lambda_0/lambda_1 = 4.07` | ROUNDED (4.07) |
| 1759 | $2.20$ | Rodgers dofs | SPEC-N:27 `degrees of freedom for signal (Rodgers): 2.20` | EXACT |
| 1763 | $\sigma_\alpha = 0.30$ | fractional prior | SPEC-A30:8 | EXACT |
| 1764 | $12.7$ | λ₀ | SPEC-A30:10 `12.73` | ROUNDED |
| 1764 | $2.99$ | λ₁ | SPEC-A30:11 `2.985` | ROUNDED |
| 1764 | $1.13$ | λ₂ | SPEC-A30:12 `1.128` | ROUNDED |
| 1764 | $0.85$ | λ₃ | SPEC-A30:13 `0.8529` | ROUNDED |
| 1764 | $0.25$ | λ₄ | SPEC-A30:14 `0.2548` | ROUNDED |
| 1764 | three combinations | count at ±30% | SPEC-A30:26 `3 of 15` | EXACT |
| 1765 | $3.4$ | Rodgers dofs at ±30% | SPEC-A30:27 `3.39` | ROUNDED |
| 1765 | same three classes leading | 23, 90, 22 | SPEC-N:10 / SPEC-A30:10–12 leading classes | EXACT |
| 1767 | one, one, two, three, four | count ladder | SPEC-A10:26 `1 of 15`; SPEC-A15:26 `1 of 15`; SPEC-A20:26 `2 of 15`; SPEC-A30:26 `3 of 15`; SPEC-A50:26 `4 of 15` | EXACT |
| 1767 | $10$, $15$, $20$, $30$, $50\%$ | the five widths | the five log filenames / their line 8 headers | EXACT |

## ¶ The pilot (lines 1774–1781)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1774 | two event hours | pilot window | RES:2375 `over 7,200 steps (2 event-hours)` | EXACT |
| 1776 | $3.03$ | pilot λ₀ | RES:1989 `lambda = 3.03, 0.64, ...`; RES:2359 `\| lambda_0 \| 3.03 \| 2.779 \|` | EXACT |
| 1776 | $2.78$ | production λ₀ | RES:2374 `It fell, 3.03 -> 2.78`; SPEC-N:10 `2.779` | ROUNDED |
| 1774 | the same count | 1 at both windows | RES:2361 `eigenvalues > 1 \| 1 \| 1` | EXACT |

## ¶ "The scan varied nearly the wrong direction" (lines 1787–1799)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1789 | $-0.64$ | v₀ on developed-medium | SPEC-N:10 `23(-0.64)` | EXACT |
| 1789 | $+0.59$ | v₀ on woody wetland | SPEC-N:10 `90(+0.59)` | EXACT |
| 1789–90 | $-0.36$ | v₀ on developed-low | SPEC-N:10 `22(-0.36)` | EXACT |
| 1790 | $-0.23$ | v₀ on pasture | SPEC-N:10 `81(-0.23)` | EXACT |
| 1792 | $0.218$ | overlap with uniform-per-class | SPEC-N:48 `\|<v_0, uniform per class>\| = 0.218` | EXACT |
| 1793 | $0.258$ | random overlap for 15 classes | RES:1998 `where random for 15 classes is 0.258` | EXACT (not recomputed; analytic value for a random unit vector would be ≈0.21, so this is the source's own convention) |
| 1794 | $0.699$ | overlap with uniform-per-cell | SPEC-N:49 `\|<v_0, uniform per cell>\| = 0.699` | EXACT |
| 1798 | $0.078$\,m | class calibration over uniform α=0.70 | RES:2075 `it wins by **0.078 m**` (0.6894 − 0.6116 = 0.0778) | EXACT |

## ¶ "What the calibration actually did" (lines 1801–1839)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1806 | $9.2$ prior std devs | calibrated displacement | RES:2047 `Total displacement 9.2 sigma (from 9.6)`; RES:2093 `displacement norm 9.16 sigma` | ROUNDED (9.16) |
| 1807 | $2.9\%$ | share in the data-constrained direction | RES:2048 `2.9% of it in the one data-constrained direction` | EXACT |
| 1808 | $85.6\%$ | share in comparable directions | RES:2049 `85.6% in comparable directions` | EXACT |
| 1808 | $11.5\%$ | share prior-determined | RES:2049 `11.5% where the prior alone decides` | EXACT |
| 1810 | $30\%$ | dev-medium narrowing, abs prior | SPEC-N:31 `23 ... learned   30%` | EXACT |
| 1810 | $28\%$ | woody wetland, abs prior | SPEC-N:32 `90 ... learned   28%` | EXACT |
| 1811 | $58\%$ | dev-medium at ±30% | SPEC-A30:31 `23 ... learned   58%` | EXACT |
| 1811 | $49\%$ | woody wetland at ±30% | SPEC-A30:32 `90 ... learned   49%` | EXACT |
| 1811 | $11$–$20\%$ | three more classes | SPEC-N:33–35 `81 20%`, `22 17%`, `21 11%` | EXACT |
| 1812 | three more | count | SPEC-N:33–35 (three rows in that band) | DERIVED |
| 1812 | $5\%$ or less … remaining ten | rest | SPEC-N:36–45 (95 5%, 24 4%, 11 2%, 31 2%, 52 1%, 42 1%, 71 1%, 82 0%, 43 0%, 41 0% = ten rows) | DERIVED/EXACT |
| 1815 | $0.739$ | deciduous forest, iteration 2 | RES:2039 `deciduous forest 0.739 -> 0.983` | EXACT |
| 1816 | $1.096$ | mixed forest, iteration 2 | RES:2040 `mixed forest 1.096 -> 1.006` | EXACT |
| 1816 | $1.274$ | cropland, iteration 2 | RES:2040 `cropland 1.274 -> 1.098` | EXACT |
| 1817 | $2\%$ or less | narrowing of those three | SPEC-N:43–45 (82 0%, 43 0%, 41 0%) | EXACT (all three are 0%; "≤2%" is a loose but true bound) |
| 1818 | iteration nine | end of the run | RES:2028 `9 total from the NLCD prior` | EXACT |
| 1819 | $10\%$ | how close they returned to the prior | DERIVED: max(\|1−0.983\|, \|1.006−1\|, \|1.098−1\|) = 9.8% | DERIVED |
| 1819 | $0.983$, $1.006$, $1.098$ | returned values | RES:2039–40; RES:2111/2113/2117 | EXACT |
| 1828 | $9.2\sigma$ | restated | RES:2047 | ROUNDED (9.16) |
| 1831 | $\alpha = 0.3$ | dev-medium bound | RES:2044 `sat at the alpha 0.3 bound`; RES:2108 `**0.300** ... (bound)` | EXACT |
| 1831 | all nine iterations | duration at the bound | RES:2044–45 `for all nine iterations` | EXACT |
| 1832 | $\alpha = 0.355$ | woody wetland | RES:2045 `woody wetland was driven to alpha 0.355`; RES:2118 `**0.355**` | EXACT |
| 1832 | $4.2\sigma$ | distance from lookup | RES:2046 `(-4.2 sigma)`; RES:2118 `-4.22` | EXACT |
| 1837 | $30\%$ or more | assigned lookup uncertainty | RES:2432 `The coauthors (2026-09-09) put the honest width at +/-30% "and higher"` | EXACT |
| 1839 | twelve | classes released by the wider prior | DERIVED: 15 − 3 supported (SPEC-A30:26) | DERIVED |

## Table tab:learned (lines 1841–1866)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1847 | 23, $30\%$, $58\%$ | developed medium | SPEC-N:31 `30%`; SPEC-A30:31 `58%` | EXACT |
| 1848 | 90, $28\%$, $49\%$ | woody wetland | SPEC-N:32 `28%`; SPEC-A30:32 `49%` | EXACT |
| 1849 | 22, $17\%$, $33\%$ | developed low | SPEC-N:34 `17%`; SPEC-A30:33 `33%` | EXACT |
| 1850 | 24, $4\%$, $25\%$ | developed high | SPEC-N:37 `4%`; SPEC-A30:34 `25%` | EXACT |
| 1851 | 81, $20\%$, $12\%$ | pasture | SPEC-N:33 `20%`; SPEC-A30:35 `12%` | EXACT |
| 1852 | 95, $5\%$, $8\%$ | emergent wetland | SPEC-N:36 `5%`; SPEC-A30:36 `8%` | EXACT |
| 1853 | 21, $11\%$, $7\%$ | developed open | SPEC-N:35 `11%`; SPEC-A30:37 `7%` | EXACT |
| 1854 | 52, 42, 43: $\le 1\%$ / $3$–$5\%$ | grouped row | SPEC-N:40,41,44 `52 1%`, `42 1%`, `43 0%`; SPEC-A30:38,39,40 `52 5%`, `42 5%`, `43 3%` | EXACT |
| 1855 | 11, 31, 41, 71, 82: $\le 2\%$ / $\le 1\%$ | grouped row | SPEC-N:38,39,45,42,43 `11 2%`, `31 2%`, `41 0%`, `71 1%`, `82 0%`; SPEC-A30:42,41,43,44,45 `11 1%`, `41 1%`, `71 1%`, `31 0%`, `82 0%` | EXACT |
| 1858 | 46 marks | survey size | SPEC-N:1 `46 marks (46 weighted)` | EXACT |
| 1861 | $\sigma_n = 0.015$ | abs prior | SPEC-N:8 | EXACT |
| 1861 | $\pm 30\%$ | fractional prior | SPEC-A30:8 | EXACT |
| 1862 | Ten classes ≤5% (absolute) | caption count | SPEC-N:36–45 (ten rows at ≤5%) | DERIVED/EXACT |
| 1863 | eight (fractional) | caption count | SPEC-A30:38–45 (52 5%, 42 5%, 43 3%, 41 1%, 11 1%, 71 1%, 31 0%, 82 0% = eight) | DERIVED/EXACT |

## ¶ "A direct test: three classes instead of fifteen" (lines 1868–1877)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1875 | three classes (23, 90, 22) | active set | RES:2130 `Calibrating ONLY classes 23, 90, 22 (chosen from the leading eigenvector)`; SPEC-N:10 leading components | EXACT |
| 1876 | the other twelve | frozen | RES:2131 `the other twelve frozen at the prior` | EXACT |
| 1877 | bit-exactly | verification of the freeze | RES:2145–46 `All twelve frozen classes sit bit-exactly at the prior in the dump (rel_err 0.000000 on every one)` | EXACT |

## Table tab:threefifteen (lines 1879–1903)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1885 | NLCD row: 0, $808.3$, $0$, $1$, $1$, $1$, $0.7188$ | lookup baseline | RES:2556 `J 894.9 vs 808.3`; RES:2068 `NLCD lookup \| 0.7188`; α ≡ 1 at the lookup by definition; no prior term at the prior centre | EXACT |
| 1886 | 15 / $\sigma_n=0.015$ / 9 its | 15-class abs-prior run | RES:2028 `9 total from the NLCD prior` | EXACT |
| 1886 | $615.4$ | J_mis | RES:2060 `J 6.153969e+02` | EXACT |
| 1886 | $41.9$ | J_prior | RES:2093 `the prior term ... is 41.94` | ROUNDED (41.94) |
| 1886 | $1.627$ | α₂₂ | RES:2107 `1.627` | EXACT |
| 1886 | $0.300$ | α₂₃ | RES:2108 `**0.300**` | EXACT |
| 1886 | $0.355$ | α₉₀ | RES:2118 `**0.355**` | EXACT |
| 1886 | $0.6116$ | MAE | RES:2060 `peak-WSE MAE 0.6116 m` | EXACT |
| 1887 | 3 / $\sigma_n=0.015$ / 4 its | 3-class abs-prior run | RES:2140 `4 TAO ... J = 6.821163e+02`; RES:2150 `converged in 4 iterations` | EXACT |
| 1887 | $648.9$ | J_mis | `logs/o61/o61_p3_score.log` `J 6.489523e+02`; RES:2317 same | EXACT |
| 1887 | $33.2$ | J_prior | RES:2154–55 `33.16` (682.12 − 648.95 = 33.17) | ROUNDED (33.16) |
| 1887 | $1.626$ | α₂₂ | RES:2164 `alpha 1.626` | EXACT |
| 1887 | $0.300$ | α₂₃ | RES:2165 `0.300 (bound)` | EXACT |
| 1887 | $0.301$ | α₉₀ | RES:2166 `0.301 (~bound)` | EXACT |
| 1887 | $0.6274$ | MAE | `logs/o61/o61_p3_score.log` `peak-WSE MAE 0.6274 m` | EXACT |
| 1888 | 15 / $\sigma_\alpha=0.30$ / 3 its | 15-class fractional run | RES:2439–40 `15 classes, sigma_alpha = 0.30, 3 TAO iterations`; `logs/o62/o62_p_c15_sa0.30.txt:1` `after 3 TAO iterations` | EXACT |
| 1888 | $615.4$ | J_mis | `logs/o62/o62_score_c15_sa0.30.log` `J 6.154375e+02` | EXACT (615.44) |
| 1888 | $16.5$ | J_prior | RES:2450 `prior term 16.5`; DERIVED 631.978 − 615.4375 = 16.54 | EXACT |
| 1888 | $1.99$ | α₂₂ | DERIVED from `o62_p_c15_sa0.30.txt` `22 0.1794520333` ÷ 0.090 = 1.994; RES:2445 `developed-low 0.179 (2x)` | DERIVED |
| 1888 | $0.300$ | α₂₃ | DERIVED `23 0.036` ÷ 0.120 = 0.300 | DERIVED |
| 1888 | $0.64$ | α₉₀ | DERIVED `90 0.06298267839` ÷ 0.098 = 0.6427; RES:2446 `woody wetland 0.063` | DERIVED |
| 1888 | $0.6154$ | MAE | `o62_score_c15_sa0.30.log` `peak-WSE MAE 0.6154 m` | EXACT |
| 1889 | 3 / $\sigma_\alpha=0.30$ / 3 its | 3-class fractional run | RES:2459 `3 (wall)`; `o62_p_c3_sa0.30.txt:1` `after 3 TAO iterations` | EXACT |
| 1889 | $633.9$ | J_mis | `o62_score_c3_sa0.30.log` `J 6.339386e+02`; RES:2459 `633.94` | EXACT |
| 1889 | $15.4$ | J_prior | DERIVED: J_tot 649.3090 (`o62_p_c3_sa0.30.txt:1`) − 633.9386 = 15.37; RES:2474 `15 (0.30)` | DERIVED |
| 1889 | $2.34$ | α₂₂ | DERIVED `22 0.2102998805` ÷ 0.090 = 2.337; RES:2459 `2.34` | EXACT |
| 1889 | $0.300$, $0.300$ | α₂₃, α₉₀ | `23 0.036` ÷ 0.120 = 0.30; `90 0.0294` ÷ 0.098 = 0.30 | DERIVED |
| 1889 | $0.6290$ | MAE | `o62_score_c3_sa0.30.log` `peak-WSE MAE 0.6290 m` | EXACT |
| 1890 | 3 / $\sigma_\alpha=0.50$ / 1 it | bracket run | `o62_p_c3_sa0.50.txt:1` `after 1 TAO iterations`; RES:2460 `1 (converged)` | EXACT |
| 1890 | $627.0$ | J_mis | `o62_score_c3_sa0.50.log` `J 6.270142e+02` | EXACT |
| 1890 | $10.0$ | J_prior | DERIVED: 636.9742 − 627.0142 = 9.96; RES:2474 `10 (0.50)` | DERIVED |
| 1890 | $3.00$ | α₂₂ | DERIVED `22 0.27` ÷ 0.090 = 3.00 | DERIVED |
| 1890 | $0.300$, $0.300$ | α₂₃, α₉₀ | as above | DERIVED |
| 1890 | $0.6295$ | MAE | `o62_score_c3_sa0.50.log` `peak-WSE MAE 0.6295 m` | EXACT |
| 1897–98 | stopped by the wall clock / $\pm 50\%$ at a corner | caption | RES:2459 `5:22 wall`, RES:2467–69 `TAO reports Residual 0 (projected gradient zero on the box) -- converged in one iteration` | EXACT |
| 1900 | $0.300$ lower bound | box floor | RES:2476 `reach the 0.3x floor` | EXACT |
| 1901 | 46 marks | MAE denominator | SPEC-N:1; score logs `dry at 0 of 46 marks` | EXACT |

## ¶ After tab:threefifteen (lines 1905–1925)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1905 | $126$ | objective units removed by three | RES:2151 `removed 126.1 of the 150.9 units` | ROUNDED (126.1); also DERIVED 808.3 − 648.9 = 159.4 J_mis — the 126.1 figure is on J_tot (808.3 − 682.1 = 126.2) |
| 1905 | $151$ | units removed by fifteen | RES:2151 `150.9` (808.3 − 657.34 = 150.96 on J_tot) | ROUNDED (150.9) |
| 1906 | $84\%$ | share of the reduction | RES:2152 `**83.6% of the reduction from 20% of the parameters**`; 126.1/150.9 = 83.6% | ROUNDED (83.6); HANDOFF:77–78 records the deliberate decision to quote 84% once |
| 1906 | $20\%$ | share of the parameters | RES:2152 `20% of the parameters` (3/15) | EXACT |
| 1907 | four iterations against nine | iteration counts | RES:2150 `4 iterations`; RES:2028 `9 total` | EXACT |
| 1910 | $0.6274$\,m | three-class MAE | `o61_p3_score.log` | EXACT |
| 1910 | $85\%$ | MAE share | RES:2324 `**85.3%**`; 0.0914/0.1072 = 85.3% | ROUNDED (85.3) |
| 1910 | $0.107$\,m | fifteen-class MAE reduction | RES:2071 `**-0.107**`; RES:2325 `-0.1072` (0.7188 − 0.6116) | EXACT |
| 1914 | $16\%$ | the remainder | DERIVED: 100% − 83.6% = 16.4% | DERIVED |
| 1915 | $8.8$ more units of prior | decomposition | RES:2154 `the 15-class run pays 8.8 MORE prior (41.94 vs 33.16)` | EXACT |
| 1916 | $33.5$ more of misfit | decomposition | RES:2155 `to buy 33.5 more misfit (615.40 vs 648.95)` | EXACT |
| 1917 | twelve non-leading classes | count | 15 − 3 | DERIVED |
| 1919 | three digits | agreement of 22/23 across runs | RES:2168 `reproduce their 15-class values to three digits` (1.626 vs 1.627) | EXACT (as stated in the source; strictly the two agree to three significant figures only in the sense the source means) |
| 1923 | $0.355$ | where the full calibration left woody wetland | RES:2118 / RES:2166 | EXACT |

## ¶ "The same three classes under the fractional prior" (lines 1927–1956)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1928 | $\sigma_\alpha = 0.30$ | prior width | RES:2433 | EXACT |
| 1929 | $0.50$ | bracket | RES:2434 `at 0.50 (task 4)` | EXACT |
| 1929 | $0.6290$ | 3-class ±30% MAE | `o62_score_c3_sa0.30.log` | EXACT |
| 1930 | $0.6295$ | 3-class ±50% MAE | `o62_score_c3_sa0.50.log` | EXACT |
| 1930 | $84\%$ | share of 0.107 m | RES:2459 `84%` (for 0.30) and RES:2460 `83%` (for 0.50) | ROUNDED — correct for the ±30% run (0.0898/0.1072 = 83.8%), but the sentence covers both and the ±50% run is 83.3% → 83% in the source. Minor over-claim on the second value. |
| 1930 | $0.107$\,m | fifteen-class reduction | RES:2462 `(c15 = the absolute-prior fifteen-class field's 0.107 m)` | EXACT |
| 1934 | seven of fifteen | classes on a bound at the first step | RES:2443 `The first step put SEVEN classes on a bound (21, 22, 81 at alpha = 3; 23, 24, 52, 90 at 0.3)` | EXACT |
| 1935 | two more iterations | dev-low relaxation | RES:2466 `then 22 relaxes 3.0 -> 2.52 -> 2.34 over two more iterations` | EXACT |
| 1936 | $\alpha = 3$ to $2.34$ | dev-low path | RES:2465–66 | EXACT |
| 1937 | one iteration ($\pm 50\%$) | stopping | RES:2469 `converged in one iteration`; `o62_p_c3_sa0.50.txt:1` | EXACT |
| 1938–39 | $15$ units at $\pm 30\%$, $10$ at $\pm 50\%$ | prior contribution | RES:2474 `the Gaussian prior contributes 15 (0.30) / 10 (0.50) J-units` | EXACT |
| 1939 | near $630$ | misfit magnitude | RES:2474 `against a misfit near 630` (633.9 and 627.0) | EXACT |
| 1940 | $\alpha \in [0.3, 3]$ | box | RES:2475 `the box [0.3, 3] does the constraining` | EXACT |
| 1943 | $\alpha = 0.3$ | dev-medium under every prior | RES:2476 `reach the 0.3x floor under every prior` | EXACT |
| 1944 | $0.355$ and $0.64$ | woody wetland when partners are free | RES:2118 `0.355`; DERIVED from `o62_p_c15_sa0.30.txt` 0.06298/0.098 = 0.643 | EXACT / DERIVED |
| 1946 | $n = 0.036$ and $0.029$ | the floor values | RES:2477 `(n = 0.036, 0.0294)`; `o62_p_c3_sa0.30.txt` `23 0.036`, `90 0.0294` | EXACT (0.0294 → 0.029) |
| 1947 | $0.040$ | lookup n for developed open | RES:2106 `21 \| developed open \| 0.040` | EXACT |
| 1948 | $1.63\times$, $2.34\times$, $3\times$ | dev-low across widths | RES:2481–82 `1.63x -> 2.34x -> 3x as the width opens` | EXACT |
| 1950 | $0.002$\,m | MAE spread over those widths | RES:2482 `MAE moves 0.002 m` (0.6274→0.6290→0.6295 spans 0.0021) | EXACT |

## ¶ "Scaling with survey size" + tab:scaling (lines 1959–2004)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1960 | 46 observations, 15 parameters | the objection | SPEC-N:1 `46 marks ..., 15 classes` | EXACT |
| 1962–63 | linear in N, $\sigma^{-2}$ | scaling law | RES:2010–11 `lambda scales linearly in observation count and as 1/sigma^2 in quality`; `plans/campaigns/o58_gauss_newton.py:8` | EXACT (property of Eq. (5)) |
| 1966 | 46 marks | this window | SPEC-N:1 | EXACT |
| 1967 | 71 … two parameters | scaling row | RECOMPUTED from SPEC-N:10–24: count of λ·(71/46) > 1 = **2** | DERIVED (note: RES:2016 gives `71 \| 1` — that older row is from the *pilot* spectrum λ₀=3.03, superseded by the production spectrum) |
| 1969 | three (108, all QC-passed) | scaling row | RECOMPUTED: λ·(108/46) > 1 count = **3** | DERIVED (RES:2017 `108 \| 2` is the superseded pilot value) |
| 1971 | five (324) | scaling row | RECOMPUTED: λ·(324/46) > 1 count = **5** | DERIVED (agrees with RES:2018) |
| 1972 | $62\%$ | Harvey marks failing QC | RES:2226 `the 122 above-bed marks` of 324 in-domain → 202/324 = 62.3% rejected | DERIVED (the paper's own line 1255 pairs the 62% with "leaving 108", which does not close: 62% of 324 leaves 122, and the 108 is after a further quality filter — an out-of-range inconsistency worth a look) |
| 1972 | 30\,m cell | mesh resolution | RES:2519 `a 30 m cell` | EXACT |
| 1973 | order $10^5$ | marks for fifteen | DERIVED, consistent with the 7×10⁵ table row | DERIVED |
| 1981 | 46 & 1 | table row | SPEC-N:26 `1 of 15` | EXACT |
| 1982 | 71 & 2 | table row | RECOMPUTED (above) | DERIVED |
| 1983 | 108 & 3 | table row | RECOMPUTED (above) | DERIVED |
| 1984 | 324 & 5 | table row | RECOMPUTED (above); RES:2018 | DERIVED/EXACT |
| 1985 | $7\times10^5$ & 15 | table row | RES:2019 `7e5 \| 15`; RECOMPUTED: the smallest eigenvalue 0.0003867 crosses 1 at N = 46/0.0003867 ≈ 1.19×10⁵, so any N ≥ ~1.2×10⁵ gives 15 | **ROUNDED / LOOSE** — 7×10⁵ is sufficient but not the threshold; the minimum N for 15 supported is ≈1.2×10⁵ (which is also what "of order 10⁵" at line 1973 says). Sourced to RES:2019, but the two statements in the same paragraph are not the same number. |
| 1989 | $\sigma = 0.15$\,m | fixed quality | SPEC-N:8 `sigma_obs = 0.15` | EXACT |
| 1990 | $\sigma_n = 0.015$ | abs prior | SPEC-N:8 | EXACT |
| 1991 | $\lambda > 1$ | supported | SPEC-N:26 | EXACT |
| 1992–93 | 46-mark row already three at ±30% | caption | SPEC-A30:26 `3 of 15` | EXACT |
| 1993 | $\sigma = 0.10$\,m buys two more (three in all) | quality scaling | RECOMPUTED: λ·(0.15/0.10)² > 1 count = **3** | DERIVED (RES:2021 says `sigma 0.10 m buys a second parameter` — the pilot value; production gives three) |
| 1994 | $\sigma = 0.05$\,m buys five | quality scaling | RECOMPUTED: λ·(0.15/0.05)² > 1 count = **5**; RES:2022 `sigma 0.05 m ... buys five` | EXACT/DERIVED (five *in all*, not five more) |
| 2011 | sixteen forward runs | restated cost | RES:1983 `16 FORWARDS` | EXACT |
| 2009 | one and three combinations | headline range | SPEC-N:26 `1 of 15`; SPEC-A30:26 `3 of 15` | EXACT |

## ¶ Sec 6.5 opening (lines 2016–2048)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2020 | 46 marks | in-sample set | SPEC-N:1 | EXACT |
| 2022 | Five (gauges) | above-bed gauges | RES:2601 `(five gauges, reconstructed exactly)`; RES:2805–09 (five rows) | EXACT |
| 2024 | 48, 48 | Katy and Houston records | RES:2600 `Katy 48 + Houston 48`; RES:2805–06 `48`, `48` | EXACT |
| 2024 | 20 | Langham Creek at Addicks | RES:2600 `Langham 20`; RES:2808 `20` | EXACT |
| 2024 | 14 | Fulshear | RES:2600 `Fulshear 14`; RES:2807 `14` | EXACT |
| 2025 | 4 | Bear Creek | RES:2600 `Bear Ck 4`; RES:2809 `4` | EXACT |
| 2025 | 134 records | total | RES:2600 `The 134 = Katy 48 + Houston 48 + Langham 20 + Fulshear 14 + Bear Ck 4`; 48+48+20+14+4 = 134 | EXACT |
| 2026 | 134 | restated | as above | EXACT |
| 2027 | $\sigma = 0.15$\,m | gauge obs error | RES:2543 `at sigma_alpha 0.30`, RES:2591 `the gauge objective used the survey grade sigma 0.15 m` | EXACT |
| 2028 | 46 marks | scoring set | `o63_score_gauge_c3_sa0.30.log` `dry at 0 of 46 marks` | EXACT |
| 2029 | one accepted quasi-Newton iteration | run length | RES:2546 `after 1 TAO iteration`; `o63_p_gauge_c3_sa0.30.txt:1` `after 1 TAO iterations` | EXACT |
| 2029 | five-hour budget | wall clock | RES:2546 `5:22 wall (exit 124 on the calibration budget)`; RES:2435 `300-min calibration budget`; RES:2598 `killed at 300 min` | EXACT |
| 2031 | two on the upper bound, third interior | active set | RES:2552 `0.27 (3.0x) \| 0.36 (3.0x) \| 0.171 (1.74x)` | EXACT |
| 2037 | $-5.3\times10^{4}$ | gauge ∂J/∂n on developed-medium | RES:2527 `dJ_gauge/dn is -53129 on developed-medium (23)`; AUDIT:97 `gauges -53129` | ROUNDED (−53,129) |
| 2038 | $+7.6\times10^{2}$ | mark ∂J/∂n on developed-medium | RES:2529 `(dJ_marks/dn +756, +675)`; AUDIT:97 `marks +756` | ROUNDED (+756) |
| 2041 | the two agree on developed-low | sign agreement | RES:2530 `They agree on developed-low (22; both want more)`; AUDIT:98 `the agreement on 22 (both negative)` | EXACT |
| 2042–43 | $+171$ predicted | mark gradient on the gauge step | AUDIT:99 `= +171 J-units`; RES:2597 `+171 predicted` | EXACT |
| 2043 | $+87$ measured | measured change | AUDIT:99 `(measured +86.7)`; RES:2597 `+86.7 measured` | ROUNDED (86.7) |
| 2044 | $0.7609$\,m | gauge field scored on marks | `logs/o63/o63_score_gauge_c3_sa0.30.log` `peak-WSE MAE 0.7609 m`; RES:2552 | EXACT |
| 2045 | $0.7188$ | uncalibrated | RES:2068, RES:2551 | EXACT |
| 2047 | $3.24$ | gauge RMSE at the lookup | RES:2510 `31313.3 \| **3.24 m**`; RES:2812 `RMSE 3.243 m overall` | EXACT |
| 2048 | $3.19$\,m | gauge RMSE of the mark field | RES:2511 `30333.4 \| 3.19 m` | EXACT |

## Table tab:crossobs (lines 2050–2073)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2056 | $0.090$, $0.120$, $0.098$ | lookup n for 22/23/90 | RES:2107–08, 2118 `0.090`, `0.120`, `0.098`; RES:2551 | EXACT |
| 2056 | $31{,}313$ | gauge J at the lookup | RES:2510 `31313.3`; RES:2621 `3.131332e+04` | EXACT |
| 2056 | $3.24$ | gauge RMSE | RES:2510 | EXACT |
| 2056 | $0.7188$ | mark MAE | RES:2068 | EXACT |
| 2057 | $0.27$, $0.36$, $0.171$ | gauge-calibrated n | `logs/o63/o63_p_gauge_c3_sa0.30.txt` `22 0.27`, `23 0.36`, `90 0.1705411928`; RES:2552 | EXACT (0.17054 → 0.171) |
| 2057 | $24{,}007$ | gauge J | `o63_p_gauge_c3_sa0.30.txt:1` `J 2.400656e+04`; RES:2592 `J_tot 24006.6` | EXACT |
| 2057 | $2.84$ | gauge RMSE | RES:2559 `Gauge RMSE 3.24 -> 2.84 m`; DERIVED 0.15·√(2·24007/134) = 2.84 | EXACT |
| 2057 | $0.7609$ | mark MAE | `o63_score_gauge_c3_sa0.30.log` | EXACT |
| 2058 | $0.210$, $0.036$, $0.029$ | mark-calibrated n | `logs/o62/o62_p_c3_sa0.30.txt` `22 0.2102998805`, `23 0.036`, `90 0.0294` | EXACT |
| 2058 | $30{,}333$ | gauge J | RES:2511 `30333.4` | EXACT |
| 2058 | $3.19$ | gauge RMSE | RES:2511 | EXACT |
| 2058 | $0.6290$ | mark MAE | `o62_score_c3_sa0.30.log` | EXACT |
| 2063 | $\sigma_\alpha = 0.30$, twelve frozen | caption | RES:2542–43 | EXACT |
| 2064 | 134 records, $\sigma = 0.15$\,m | caption | RES:2600, RES:2591 | EXACT |
| 2065 | RMSE $=\sigma\sqrt{2J/134}$ | caption formula | RES:2506 `RMSE = 0.15 sqrt(2J/134)` | EXACT |
| 2066 | 46 cluster-A marks | caption | SPEC-N:1; RES:2015 `the 12-hr cluster-A window` | EXACT |
| 2068 | $0.27$ and $0.36$ at the $\alpha = 3$ bound | caption | DERIVED: 0.27/0.090 = 3.0, 0.36/0.120 = 3.0; RES:2552 `(3.0x)`, `(3.0x)` | DERIVED/EXACT |

## ¶ "What the gauges measure" (lines 2075–2090)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2080 | $99.5\%$ | offset share of J | RES:2811 `J 31313 = 31145 constant per-gauge offset (99.5%) + 168 hydrograph shape (0.54%)` | EXACT |
| 2081 | $4.9$\,m | Houston mean residual | RES:2805 `**+4.901 m**`; RES:2637 `**+4.90 m**` | EXACT |
| 2081–82 | $82\%$ | Houston share of misfit | RES:2805 `**82.3%**` | ROUNDED (82.3) |
| 2082 | $1$–$2$\,m at three of the other four | other gauges | RES:2806–09 `+1.944`, `+1.965`, `+1.197` (and Langham `-0.613`) | DERIVED/EXACT |
| 2082 | $0.24$\,m | shape RMSE | RES:2812 `**0.238 m in shape alone**` | ROUNDED (0.238) |
| 2083 | $3.24$\,m | total RMSE | RES:2812 `RMSE 3.243 m overall` | EXACT |
| 2086 | three times the lookup on two developed classes | the tested change | `o63_p_gauge_c3_sa0.30.txt` 22 and 23 at 3.0× | EXACT |
| 2087 | $0.4$\,m of the $3.24$ | removed by that field | DERIVED: 3.24 − 2.84 = 0.40 (RES:2559) | DERIVED |
| 2088 | $2.8$\,m | what remains | DERIVED: the 2.84 m post-step RMSE (RES:2559) | DERIVED |

## Table tab:gaugeresid (lines 2092–2111)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2098 | Houston: 48, $+4.90$, $0.38$, $82\%$ | gauge row | RES:2805 `48 \| **+4.901 m** \| 0.382 m \| **82.3%**` | ROUNDED (4.901, 0.382, 82.3) |
| 2099 | Katy: 48, $+1.94$, $0.07$, $13\%$ | gauge row | RES:2806 `48 \| +1.944 m \| 0.067 m \| 12.9%` | ROUNDED (1.944, 0.067, 12.9) |
| 2100 | Fulshear: 14, $+1.97$, $0.06$, $4\%$ | gauge row | RES:2807 `14 \| +1.965 m \| 0.055 m \| 3.8%` | ROUNDED (1.965, 0.055, 3.8) — the shape value 0.055 rounds to 0.06 only by rounding-half-up; 3.8% → 4% |
| 2101 | Langham: 20, $-0.61$, $0.12$, $0.6\%$ | gauge row | RES:2808 `20 \| -0.613 m \| 0.123 m \| 0.6%` | ROUNDED/EXACT |
| 2102 | Bear Ck: 4, $+1.20$, $0.01$, $0.4\%$ | gauge row | RES:2809 `4 \| +1.197 m \| 0.012 m \| 0.4%` | ROUNDED/EXACT |
| 2106 | 134 above-bed records | caption | RES:2600 | EXACT |
| 2108 | $99.5\%$ | caption | RES:2811 | EXACT |
| 2109 | $0.5\%$ | shape share | RES:2811 `168 hydrograph shape (0.54%)` | ROUNDED (0.54) |
| 2109 | $3.24$\,m, $0.24$\,m | caption | RES:2812 `RMSE 3.243 m overall, **0.238 m in shape alone**` | ROUNDED |

## ¶ Final paragraphs (lines 2113–2146)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2114 | $3.24$ to $2.84$\,m | gauge-only reported reduction | RES:2559 `Gauge RMSE 3.24 -> 2.84 m` | EXACT |
| 2115–16 | $0.27$ and $0.36$ | gauge-calibrated roughness | `o63_p_gauge_c3_sa0.30.txt` | EXACT |
| 2125–26 | Tables ladder / threefifteen / crossobs as achieved fits | labelling | HANDOFF:194–197 (`the prior's 0.7188 m; 0.6116 / 0.6154 / 0.6274 / 0.6290 ... tab:threefifteen`) | EXACT (label, not a number) |
| 2140–41 | two cross-scores each worse than the lookup | claim | RES:2552–53 (0.7609 > 0.7188 on marks; 30333 vs 31313 on gauges is *better*, see note) | **PARTIALLY CONTRADICTED** — the mark-calibrated field is *better* than the lookup on the gauges (30,333 vs 31,313; RMSE 3.19 vs 3.24), so "each is worse than the uncalibrated lookup on the other observable" holds only for the gauge-calibrated field. |
| 2141 | $0.090$\,m | three-class improvement at the marks | DERIVED: 0.7188 − 0.6290 = 0.0898; RES:2459 `-0.090` | DERIVED/EXACT |
| 2143 | three-combination count | restated | SPEC-A30:26 `3 of 15` | EXACT |

---

## NOT FOUND

None. Every number in the range traced to a log, a RESULTS/audit entry, or a stated derivation.

## CONTRADICTED / needs an edit

1. **line 1722, `107`** — the count belongs to the earlier 14-column analysis (RES:2383 `151 of 644`, RES:2387 `107 pairs (71%)`). Recomputed on the full 15-column set the paper actually cites (156 of 690): **112 pairs shift by one 300-step sample (71.8%)**. The `71%` survives; the `107` should be `112`.
2. **line 1726, `1.6\%`** — 10/644 = 1.6% on the old sample; against the paper's own 690 it is **1.4%**.
3. **line 1726, "relocate by more than an hour"** — the source bin is 2100–6900 steps (RES:2389). At dt = 1 s an hour is 3600 steps; recomputed, only **7 of the 10** exceed an hour (two shift 2100 steps = 35 min, one 3300 = 55 min). Either say "by more than half an hour" for ten, or "seven pairs by more than an hour".
4. **line 1670, `35\%`** — sourced (AUDIT:66, HANDOFF:30) but it is the **gauge** objective along the gauge step (−7354 measured vs −21283 predicted). The sentence points the reader to Sec 6.5, where the only measured-vs-predicted pair printed is +171 / +87 = **51%**. Either quote 51% with the Sec 6.5 pointer, or state that the 35% is the gauge objective's own saturation and cite the audit.
5. **line 1985, `7\times10^5`** — sourced to RES:2019 but it is not the threshold: λ₁₅ = 0.0003867 crosses unity at N ≈ **1.2×10⁵**, which is what line 1973's "of order $10^5$" says. The table row and the prose disagree by a factor of six.
6. **lines 2140–41, "each is worse than the uncalibrated lookup on the other observable"** — true of the gauge-calibrated field (0.7609 vs 0.7188 at the marks) but **false of the mark-calibrated field**, which improves the gauges (J 30,333 vs 31,313; RMSE 3.19 vs 3.24, RES:2511). The paper itself prints the improving numbers at line 2048 and in tab:crossobs.
7. **line 1930, `84\%`** — correct for the ±30% run (83.8%) but the sentence covers the ±50% run too, which RES:2460 scores at **83%**.
8. **line 1972, `62\%`** (context only; the arithmetic lives at out-of-range line 1255) — 62% is the *above-bed* rejection rate of the 324 in-domain marks (324 → 122, RES:2226), not the rate that leaves 108. The 108 follows a further quality filter.
