# Provenance trace D — manning-calibration.tex lines 1366–1646 and 2147–2247

Abbreviations for sources:
- `RES` = `/Users/markadams/Codes/RDycore-gpu/plans/RESULTS-gpu-implicit.md`
- `PS` = `/Users/markadams/Codes/RDycore-gpu/plans/PROJECT-STATE.md`
- `PS26` = `/Users/markadams/Codes/RDycore-gpu/plans/PROJECT-STATE-2026-08-26.md`
- `TDL` = `/Users/markadams/Codes/RDycore-gpu/plans/team-decision-list.md`
- log paths are under `/Users/markadams/Codes/RDycore-gpu/logs/`
- campaign scripts under `/Users/markadams/Codes/RDycore-gpu/plans/campaigns/`

Note on raw logs: **no raw logs exist in-repo for o44/o45 (alpha scan), o47
(halves), o49 (alpha 0.70/0.80), or o54 (IC scan)** — `find . -name '*o44*' …`
returns only `plans/campaigns/o49_alpha_bar.sh` and
`plans/campaigns/o54_ic_authority.sh`. Those four campaigns are sourced only by
the running results record (`RES`), which is provenance source (2). Rows below
marked EXACT against `RES` are therefore one level removed from a raw log.

---

## Sec 6.2 opening paragraphs (lines 1369–1406)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1369 | fifteen | land-cover classes to fit | `RES:2103–2119` full 15-row NLCD class table | EXACT |
| 1384 | 12-hour | production window | `RES:1572` "12-hr cluster-A window h29-41"; `PS:97` | EXACT |
| 1385 | hours 29--41 | upstream crest cluster | `RES:1573` "window h29-41"; `PS:104` | EXACT |
| 1386 | 72-hour | forward the checkpoint comes from | `PS:97` "restart from a 72-hour forward"; `RES:1287` | EXACT |
| 1387 | 46 | surveyed marks scored | `RES:1573` "46 real marks"; `RES:1582` "0/46" | EXACT |
| 1396 | 29--42 and 60--72 | bimodal crest clusters | `PS:111` "crests here are bimodal (h29–42 and h60–72)"; `plans/meeting-summary-2026-08-26.md:38` | EXACT |
| 1401 | 46 | marks used throughout | as line 1387 | EXACT |

---

## tab:alpha (lines 1408–1435)

`n` ranges are `alpha` × the lookup's min 0.027 (barren) and max 0.160
(developed high), `RES:2110`, `RES:2109`.

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1414 | $0.20$ | alpha row | `RES:1578` "\| 0.2 \|" | EXACT |
| 1414 | $0.005$--$0.032$ | n range at 0.20 | `RES:1578` "0.005-0.032" | EXACT |
| 1414 | implicit solve diverges | J cell | `RES:1578` "**DIVERGED_NONLINEAR_SOLVE**" | EXACT |
| 1415 | $0.30$ | alpha row | `RES:1579` | EXACT |
| 1415 | $0.008$--$0.048$ | n range | `RES:1579` "0.008-0.048" | EXACT |
| 1415 | $6.737\times10^{2}$ | J | `RES:1579` "6.7367e2"; `RES:1899` "673.67" | ROUNDED (673.67) |
| 1415 | $0.6392$ | MAE | `RES:1579` "**0.6392**" | EXACT |
| 1415 | 1/46 | dry | `RES:1592` "1/46 dry throughout" | EXACT |
| 1416 | $0.45$ | alpha row | `RES:1580` | EXACT |
| 1416 | $0.012$--$0.072$ | n range | `RES:1580` "0.012-0.072" | EXACT |
| 1416 | $7.038\times10^{2}$ | J | `RES:1580` "7.0380e2"; `RES:1899` "703.80" | EXACT |
| 1416 | $0.6614$ | MAE | `RES:1580` "0.6614" | EXACT |
| 1416 | 1/46 | dry | `RES:1592` "1/46 dry throughout" | EXACT |
| 1417 | $0.60$ | alpha row | `RES:1581` | EXACT |
| 1417 | $0.016$--$0.096$ | n range | `RES:1581` "0.016-0.096" | EXACT |
| 1417 | $7.406\times10^{2}$ | J | `RES:1581` "7.4056e2"; `RES:1899` "740.56" | ROUNDED (740.56) |
| 1417 | $0.6776$ | MAE | `RES:1581` "0.6776" | EXACT |
| 1417 | 1/46 | dry | `RES:1592` "1/46 dry throughout" | EXACT |
| 1418 | $0.70$ | alpha row | `RES:1897`, `campaigns/o49_alpha_bar.sh:29` | EXACT |
| 1418 | $0.019$--$0.112$ | n range | not tabulated; 0.70 × 0.027 = 0.0189 → 0.019; 0.70 × 0.160 = 0.112 (`RES:2110`, `RES:2109`) | DERIVED |
| 1418 | $7.655\times10^{2}$ | J | `RES:2265` "J 7.654816e2"; `RES:1899` "765.48" | ROUNDED (765.4816) |
| 1418 | $0.6894$ | MAE | `RES:1900`, `RES:2265` | EXACT |
| 1418 | 1/46 | dry | `RES:2264` "`dry at 1 of 46` at alpha = 0.70" | EXACT |
| 1419 | $0.80$ | alpha row | `RES:1897` | EXACT |
| 1419 | $0.022$--$0.128$ | n range | 0.80 × 0.027 = 0.0216 → 0.022; 0.80 × 0.160 = 0.128 | DERIVED |
| 1419 | $7.813\times10^{2}$ | J | `RES:2266` "J 7.812901e2"; `RES:1899` "781.29" | ROUNDED (781.2901) |
| 1419 | $0.6991$ | MAE | `RES:1900`, `RES:2266` | EXACT |
| 1419 | 0/46 | dry | `RES:2265–2266` "`0 of 46` at alpha = 0.80" | EXACT |
| 1420 | $1.00$ (NLCD) | alpha row | `RES:1582` | EXACT |
| 1420 | $0.027$--$0.160$ | n range | `RES:1582` "0.027-0.160" | EXACT |
| 1420 | $8.083\times10^{2}$ | J | `RES:1582` "8.0826e2"; `logs/o59/o59_57649525.log:84` "J 8.082566e+02" | ROUNDED (808.2566) |
| 1420 | $0.7188$ | MAE | `RES:1582`; `logs/o59/o59_57649525.log:84` | EXACT |
| 1420 | 0/46 | dry | `RES:1582` "0/46"; `logs/o59/o59_57649525.log:84` "dry at 0 of 46" | EXACT |
| 1424 | 12-hour, 46 | caption config | as above | EXACT |
| 1428 | $0.08$\,m | worth of whole uniform scale | `RES:1584` "worth 0.08 m"; also 0.7188 − 0.6392 = 0.0796 | EXACT |
| 1428 | $0.72$\,m | uncalibrated error | `RES:1584` "against a 0.72 m error" (0.7188 rounded) | EXACT |
| 1429 | $30\%$ | alpha at which it is reached | `RES:1585` "roughness at 30% of the published NLCD values" | EXACT |
| 1429 | $0.008$ | lowest n reached | `RES:1586` "n as low as 0.008" | EXACT |
| 1430 | $\approx 0.012$ | smooth concrete | `RES:1586` "smooth concrete is ~0.012" | EXACT |
| 1431 | $3$\,cm | worth within the prior | `RES:1588` "worth about 3 cm"; also 0.7188 − 0.6894 = 0.0294 | EXACT |
| 1431 | $\alpha = 0.70$ | prior-implied optimum | `RES:1903` "predicted optimum for sigma_n = 0.015 was alpha 0.697 at MAE 0.6895; the measured point at 0.70 is 0.6894" | EXACT |
| 1432 | $\alpha = 0.2$ | numerical limit | `RES:1578`, `RES:1592–1593` "at alpha 0.2 the implicit solve diverges" | EXACT |
| 1433 | one mark stays dry throughout | dry claim | `RES:1592` "The limit is NOT drying (1/46 dry throughout)" — but that statement covers only the o44/o45 alphas (0.3/0.45/0.6); the table's own 0.80 and 1.00 rows are 0/46 | CONTRADICTED by the table's own rows (see note) |

**Note on line 1433**: the source sentence `RES:1592` is scoped to the three
o44/o45 sub-unity alphas. As printed in the paper the caption sits under a table
whose last two rows read 0/46, so "throughout" is false of the table it captions.
Suggest "one mark stays dry below $\alpha = 0.8$".

---

## fig:authority caption (lines 1437–1452)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1442 | 46 | marks on both curves | `RES:1573` | EXACT |
| 1443 | $1.0$ | both curves pass through here | `RES:1582` (alpha 1.0), `RES:1942` (a 1.00) — both are the same 0.7188 point | EXACT |
| 1445 | $n = 0.008$ | useful end of roughness | `RES:1586` | EXACT |
| 1446 | about four times steeper | IC vs roughness slope | `RES:1936` "~4x the authority of roughness"; 0.392 / 0.098 = 4.00 (`RES:1945–1946`) | DERIVED/EXACT |
| 1448 | $40\%$ reduction | IC not turned over | `RES:2257` a = 0.6 → MAE 0.5818; `PS:68` "antecedent water −40%" | EXACT |
| 1448 | fifteen-class | dotted line | `RES:2071` "calibrated, 15 classes, 9 its — 0.6116" | EXACT |

---

## Sec 6.2 prose after the figure (lines 1454–1469)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1454 | $0.08$\,m | whole uniform scale | `RES:1584` | EXACT |
| 1455 | $0.72$\,m | model--survey error | `RES:1584` | EXACT |
| 1455 | $11\%$ | fraction of the error | `RES:1584–1585` "(11%)"; 0.0796 / 0.7188 = 11.1% | EXACT |
| 1456 | $\alpha = 0.3$ | where it is reached | `RES:1585` | EXACT |
| 1456 | $30\%$ | of published NLCD | `RES:1585` | EXACT |
| 1457 | $n = 0.008$ | lowest n | `RES:1586` | EXACT |
| 1458 | $\approx 0.012$ | smooth concrete | `RES:1586` | EXACT |
| 1458 | $\alpha \gtrsim 0.7$ | inside the prior | `RES:1588` "alpha >~ 0.7" | EXACT |
| 1459 | three centimetres | worth inside the prior | `RES:1588` "about 3 cm" | EXACT |
| 1465 | $\sim 0.1$\,m | twin's peak-WSE signal | `RES:1596` "peak-WSE signal at ~0.1 m for a large class perturbation" | EXACT |
| 1466 | $85\%$ or more | residual that is not roughness | `RES:1597` "~85-90% of the model-vs-survey residual is NOT roughness" | EXACT (paper's "or more" covers the 85–90 band) |
| 1469 | 30\,m | representation-error scale | `RES:1599` "representation error at 30 m" | EXACT |

---

## tab:halves and surrounding prose (lines 1471–1517)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1471 | fifteen / one | modes constrained | `RES:1641–1644` | EXACT |
| 1475 | $\alpha = 0.45$ | half-scan scale | `RES:1620` "Half-scans at alpha 0.45" | EXACT |
| 1484 | $808.3$ | NLCD prior J | `RES:1625` "808.3" | EXACT |
| 1484 | $0.7188$ | NLCD prior MAE | `RES:1625` | EXACT |
| 1485 | $71.6\%$ | developed 21–24 share of cells | `RES:1626` "(71.6% of cells)" | EXACT |
| 1485 | $786.8$ | developed-only J | `RES:1626` | EXACT |
| 1485 | $-21.5$ | ΔJ | `RES:1626` "-21.5"; 786.8 − 808.3 = −21.5 | EXACT |
| 1485 | $0.7306$ | developed-only MAE | `RES:1626` | EXACT |
| 1485 | $+0.0118$ | ΔMAE | `RES:1626` "**+0.0118**"; 0.7306 − 0.7188 = 0.0118 | EXACT |
| 1486 | $28.4\%$ | everything else | `RES:1627` "(28.4%)"; also 100 − 71.6 | EXACT |
| 1486 | $827.5$ | J | `RES:1627` | EXACT |
| 1486 | $+19.2$ | ΔJ | `RES:1627`; 827.5 − 808.3 = 19.2 | EXACT |
| 1486 | $0.7268$ | MAE | `RES:1627` | EXACT |
| 1486 | $+0.0080$ | ΔMAE | `RES:1627` "**+0.0080**"; 0.7268 − 0.7188 = 0.0080 | EXACT |
| 1487 | $703.8$ | both / uniform 0.45 J | `RES:1628`; `RES:1899` "703.80" | EXACT |
| 1487 | $-104.5$ | ΔJ | `RES:1628` "**-104.5**"; 703.8 − 808.3 = −104.5 | EXACT |
| 1487 | $0.6614$ | MAE | `RES:1628`, `RES:1580` | EXACT |
| 1487 | $-0.0574$ | ΔMAE | `RES:1628` "**-0.0574**"; 0.6614 − 0.7188 = −0.0574 | EXACT |
| 1491 | $45\times$ | sum-of-parts ratio (caption) | `RES:1630` "together they help by 45x the sum of the parts"; in J: −104.5 / (−21.5 + 19.2) = 45.4 | EXACT / DERIVED |
| 1498 | $45\times$ | same, in prose | `RES:1630` | EXACT |
| 1507 | \emph{one} | roughness d.o.f. at the absolute prior width | `logs/o61/o61_spectrum_sigma_n0.015.txt:26` "eigenvalues > 1 : 1 of 15"; `RES:2415` | EXACT |
| 1510 | $0.08$\,m | worth of that mode | `RES:1643` "worth ~0.08 m" | EXACT |
| 1510 | $0.72$\,m | discrepancy | `RES:1644` "against a 0.72 m error" | EXACT |
| 1511 | about three | linearized count | `logs/o61/o61_spectrum_sigma_alpha0.30.txt:26` "eigenvalues > 1 : 3 of 15" | EXACT |
| 1512 | $\pm 30\%$ | prior width for that count | same file header "sigma_alpha = 0.3 uniform" | EXACT |
| 1512 | fifteen | classes' worth of information | `RES:1644` | EXACT |

---

## "Relation to equifinality" (lines 1519–1531)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1522 | beven1992glue, beven2006equifinality | equifinality thesis | citation | N/A-citation |
| 1527 | four forward evaluations | bounding roughness authority | `campaigns/o49_alpha_bar.sh:15` "o44/o45 measured 0.3, 0.45, 0.6, 1.0"; `RES:1576–1582` (the 0.2 row diverged and is not an evaluation) | DERIVED (4 completed forwards) |
| 1528 | two more | show modes not separable | `RES:1626–1627` — the two new half-domain fields; row 1 (NLCD) and row 4 (uniform 0.45) were already run | DERIVED |

---

## Sec 6.3 opening (lines 1544–1553)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1551 | fifteen | NLCD classes calibrated | `RES:2103–2119` | EXACT |
| 1551 | both prior widths | σ_n 0.015 and σ_α 0.30 | `RES:2439–2441` | EXACT |
| 1553 | 46 | marks scored | `RES:2058` | EXACT |

---

## tab:ladder (lines 1555–1589)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1561 | $0.7188$ | NLCD lookup MAE | `RES:2068` | EXACT |
| 1562 | $\alpha = 0.70$, $0.6894$ | uniform prior-consistent optimum | `RES:2069` | EXACT |
| 1562 | $-0.029$ | Δ | `RES:2069` "-0.029"; 0.6894 − 0.7188 = −0.0294 | EXACT |
| 1562 | $0.70$ of table | where the field sits | `RES:2069` | EXACT |
| 1563 | $\alpha = 0.30$, $0.6392$ | unregularized floor | `RES:2070` | EXACT |
| 1563 | $-0.080$ | Δ | `RES:2070` "-0.080"; 0.6392 − 0.7188 = −0.0796 | EXACT |
| 1563 | $0.008$ | n down to | `RES:2070` "no, n to 0.008" | EXACT |
| 1564 | 15 classes, $\sigma_n = 0.015$, $0.6116$ | headline calibration | `RES:2060` "peak-WSE MAE 0.6116 m"; `RES:2071` | EXACT |
| 1564 | $-0.107$ | Δ | `RES:2071` "**-0.107**"; 0.6116 − 0.7188 = −0.1072 | EXACT |
| 1564 | $0.036$ | developed-medium on the floor | `RES:2108` "23 \| developed medium \| 0.120 \| 0.03600 \| 0.300 (bound)" | EXACT |
| 1565 | 15 classes, $\sigma_\alpha = 0.30$, $0.6154$ | wider-prior control | `RES:2440` "**MAE 0.6154 m, J_mis 615.44**" | EXACT |
| 1565 | $-0.103$ | Δ | 0.6154 − 0.7188 = −0.1034 | DERIVED |
| 1565 | $0.036$ (developed-medium) | class value | `RES:2444` "developed-medium 0.036 … on the floor" | EXACT |
| 1565 | $0.035$ (shrub) | class value | `RES:2444` "shrub 0.0345 on the floor" | ROUNDED (0.0345) |
| 1566 | 3 classes, $\sigma_n = 0.015$, $0.6274$ | o59/o61 three-class field | `RES:2317` "peak-WSE MAE 0.6274 m"; `RES:2458` | EXACT |
| 1566 | $-0.091$ | Δ | `RES:2458` "-0.091"; 0.6274 − 0.7188 = −0.0914 | EXACT |
| 1566 | $0.036$ (developed-medium) | class value | `RES:2477` "n = 0.036, 0.0294" | EXACT |
| 1566 | $0.029$ (woody wetland) | class value | `RES:2477` "0.0294"; `RES:2553` "0.029 (0.30x)" | ROUNDED (0.0294) |
| 1567 | 3 classes, $\sigma_\alpha = 0.30$, $0.6290$ | o62 task 3 | `RES:2459` "**0.6290**" | EXACT |
| 1567 | $-0.090$ | Δ | `RES:2459` "-0.090"; 0.6290 − 0.7188 = −0.0898 | EXACT |
| 1567 | $0.21$ (developed-low) | class value | `RES:2553` "0.210 (2.34x)" | EXACT |
| 1567 | $2.3\times$ | developed-low multiple | `RES:2459` "2.34"; `RES:2553` "(2.34x)" | ROUNDED (2.34) |
| 1568 | 3 classes, $\sigma_\alpha = 0.50$, $0.6295$ | o62 task 4 | `RES:2460` "0.6295" | EXACT |
| 1568 | $-0.089$ | Δ | `RES:2460` "-0.089"; 0.6295 − 0.7188 = −0.0893 | EXACT |
| 1568 | $0.27$ (developed-low) | class value | `RES:2465` "22 -> 0.27 = 3x" | EXACT |
| 1568 | $3\times$ | developed-low multiple | `RES:2460` "3.00"; `RES:2465` "= 3x" | EXACT |
| 1571 | 46 | cluster-A marks | `RES:2058` | EXACT |
| 1574 | $\alpha = 0.70$ | best uniform the prior admits | `RES:1903` | EXACT |
| 1574 | $0.078$\,m | margin over α = 0.70 | `RES:2075` "it wins by **0.078 m**"; `RES:2342`; 0.6894 − 0.6116 = 0.0778 | EXACT |
| 1576 | three and a half times | redistribution vs global scale | `RES:2075–2076` "~3.6x what the best defensible uniform field reaches"; 0.1072 / 0.0294 = 3.65 | ROUNDED (3.6×) |
| 1576 | nine | quasi-Newton its at σ_n = 0.015 | `RES:2028` "9 total from the NLCD prior"; `RES:2071` "9 its" | EXACT |
| 1577 | three | its at σ_α = 0.30 | `RES:2439` "3 TAO iterations" | EXACT |
| 1578 | four figures | agreement in misfit | `RES:2440–2442` 615.44 vs 615.40 | EXACT |
| 1578 | $J_{\rm mis} = 615.4$ | shared misfit floor | `RES:2060` "J 6.153969e+02"; `RES:2462` "J_mis 615.4" | ROUNDED (615.3969 / 615.44) |
| 1580 | $0.027$ (barren) | lookup minimum | `RES:2110` "31 \| barren \| 0.027" | EXACT |
| 1581 | $0.160$ (developed high) | lookup maximum | `RES:2109` "24 \| developed high \| 0.160" | EXACT |
| 1581 | $0.040$ | lowest developed entry, developed open | `RES:2106` "21 \| developed open \| 0.040" | EXACT |
| 1582 | $n = 0.036$ | developed-medium calibrated | `RES:2108` "0.03600" | EXACT |
| 1582 | $0.029$ | woody wetland calibrated (3-class) | `RES:2477` "0.0294" | ROUNDED (0.0294) |

---

## Sec 6.3 prose (lines 1591–1646)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1592 | $15\%$ | roughness share of the error | `RES:2073` "Roughness accounts for **15%** of the 0.72 m discrepancy"; 0.1072 / 0.7188 = 14.9% | EXACT |
| 1592 | $0.72$\,m | model--survey error | `RES:2073` | EXACT |
| 1594 | 46 / 46 | in-sample marks | `RES:2058` | EXACT |
| 1595 | $85\%$ | remaining error | 100 − 15 (`RES:2073`); cf. `RES:1597` "~85-90%" | DERIVED |
| 1596 | 30\,m | representation error scale | `RES:1599` | EXACT |
| 1600 | $15\%$ | repeat | `RES:2073` | EXACT |
| 1601 | $\alpha = 0.70$ / $\alpha = 0.30$ | competitor choice | `RES:2074` "Against the honest one-parameter competitor (alpha 0.70)" | EXACT |
| 1602 | $n = 0.008$ | α = 0.30 floor | `RES:2070` | EXACT |
| 1605 | $0.078$\,m | calibration's margin | `RES:2075` | EXACT |
| 1614 | $J = 615.4$ | calibrated misfit | `RES:2060` "6.153969e+02"; `RES:2086` "J 615.4" | ROUNDED (615.3969) |
| 1616 | $\approx 0.596$\,m | MAE the uniform relation extrapolates to | `RES:2084` "gives 0.596 against the calibrated 0.6116"; slope `RES:2083` dMAE/dJ = 7.37e-4 from the 0.30/0.45 points | EXACT |
| 1617 | $0.6116$ | calibrated MAE | `RES:2060` | EXACT |
| 1620 | $J = 615.4$ | repeat | `RES:2086` | EXACT |
| 1620 | $673.7$ | lowest uniform objective before divergence | `RES:2086` "58 units BELOW 673.7"; `RES:1899` "673.67" (α = 0.30) | ROUNDED (673.67) |
| 1624 | nine | quasi-Newton iterations | `RES:2028` | EXACT |
| 1625 | 12-hour wall | stopping cause | `RES:2027–2028` "12-hr slot, stopped by the wall after 7 more iterations"; `RES:2051` | EXACT |
| 1626 | $18.7\%$ | total J reduction | `RES:2034` "an 18.7% total reduction"; 808.3 → 657.3 = −18.68% | EXACT |
| 1627 | $0.01\%$ | last-iteration J gain | `RES:2035` "(the last iteration gained 0.01%)" | EXACT |
| 1627 | $8.5\times$ | gradient-norm fall | `RES:2030–2032` "\|g\| 0.135663 → 0.0159113 ← gradient norm fell 8.5x"; 0.135663 / 0.0159113 = 8.53 | EXACT |
| 1631 | $15\%$ | repeat | `RES:2073` | EXACT |
| 1632 | $n = 0.036$ | developed-medium lower bound | `RES:2108`; `RES:2044` "sat at the alpha 0.3 bound for all nine iterations" | EXACT |
| 1633 | $0.040$ | table's developed-open entry | `RES:2106` | EXACT |
| 1635 | seven of the fifteen | classes bounded on the first step | `RES:2443` "The first step put SEVEN classes on a bound (21, 22, 81 at alpha = 3; 23, 24, 52, 90 at 0.3)" | EXACT |
| 1637 | $9.2$ | prior standard deviations of displacement | `RES:2047` "Total displacement 9.2 sigma (from 9.6)"; `campaigns/o59_spectral_active_set.sh:25` | EXACT |
| 1638 | $2.9\%$ | share in a constrained direction | `RES:2048` "with 2.9% of it in the one data-constrained direction"; `campaigns/o59_spectral_active_set.sh:26` | EXACT |

---

## Sec 7 opening (lines 2147–2167)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2150 | $15\%$ | roughness share | `RES:2073` | EXACT |
| 2152 | $85\%$ | the rest | 100 − 15 | DERIVED |
| 2162 | hour 29 | window start | `PS:104`; `RES:1573` | EXACT |
| 2163 | 72-hour | forward the checkpoint comes from | `PS:97` | EXACT |
| 2164 | three conserved variables | h, hu, hv scaled by a | `RES:1937–1938` "Scaling the whole restart state by a (velocities unchanged)"; `RES:1960` "~8.8M unknowns" = 3 × 2.93M | EXACT |

---

## tab:ic (lines 2169–2188)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2175 | $0.80$ | IC scale | `RES:1940` | EXACT |
| 2175 | $6.652\times10^{2}$ | J | `RES:1942` "665.17"; `campaigns/o60_ic_extend.sh:82` "J 6.652e+02" | ROUNDED (665.17) |
| 2175 | $0.6434$ | MAE | `RES:1943`; `campaigns/o60_ic_extend.sh:82` | EXACT |
| 2176 | $0.90$ | IC scale | `RES:1940` | EXACT |
| 2176 | $7.284\times10^{2}$ | J | `RES:1942` "728.43"; `campaigns/o60_ic_extend.sh:81` "J 7.284e+02" | ROUNDED (728.43) |
| 2176 | $0.6796$ | MAE | `RES:1943`; `campaigns/o60_ic_extend.sh:81` | EXACT |
| 2176 | $+0.0362$ | ΔMAE per 0.1 | 0.6796 − 0.6434 = 0.0362 | DERIVED |
| 2177 | $1.00$ (checkpoint) | IC scale | `RES:1940` | EXACT |
| 2177 | $8.083\times10^{2}$ | J | `RES:1942` "808.26"; `campaigns/o54_ic_authority.sh:91` "J 8.082566e+02" | ROUNDED (808.2566) |
| 2177 | $0.7188$ | MAE | `RES:1943` | EXACT |
| 2177 | $+0.0392$ | ΔMAE | 0.7188 − 0.6796 = 0.0392 | DERIVED |
| 2178 | $1.10$ | IC scale | `RES:1940` | EXACT |
| 2178 | $9.071\times10^{2}$ | J | `RES:1942` "907.11"; `campaigns/o60_ic_extend.sh:79` "J 9.071e+02" | ROUNDED (907.11) |
| 2178 | $0.7610$ | MAE | `RES:1943`; `campaigns/o60_ic_extend.sh:79` | EXACT |
| 2178 | $+0.0422$ | ΔMAE | 0.7610 − 0.7188 = 0.0422 | DERIVED |
| 2179 | $1.20$ | IC scale | `RES:1940` | EXACT |
| 2179 | $1.025\times10^{3}$ | J | `RES:1942` "1024.51"; `campaigns/o60_ic_extend.sh:78` "J 1.025e+03" | ROUNDED (1024.51) |
| 2179 | $0.8174$ | MAE | `RES:1943`; `campaigns/o60_ic_extend.sh:78` | EXACT |
| 2179 | $+0.0564$ | ΔMAE | 0.8174 − 0.7610 = 0.0564 | DERIVED |
| 2184 | about $1\%$ | straightness over [0.8, 1.1] | `RES:1952–1954` "straight to about 1% between 0.8 and 1.1 and CONVEX above (second differences 0.0030, 0.0030, 0.0142)" | EXACT |
| 2184 | $a = 0.8$ and $1.1$ | straight range | `RES:1952` | EXACT |
| 2186 | hour 29 | window start | `RES:1957` | EXACT |
| 2186 | No mark is dry at any point | dry claim across the IC scan | only a = 1.0 (`campaigns/o54_ic_authority.sh:91` "0/46 dry"), a = 0.7 and a = 0.6 (`logs/o57/o57_ic0.6.log:81`, `logs/o60/o60_ic0.6.log:81`, `RES:2286–2287`) are documented. Grepped `0.6796`, `0.7610`, `0.8174`, `665.17`, `728.43`, `907.11`, `1024.51`, `"dry at"` across `plans/` and `logs/`; no o54 raw log exists in-repo (`find . -name '*o54*'` returns only the campaign script) | **NOT FOUND** for a = 0.8, 0.9, 1.1, 1.2 |

---

## Sec 7 prose (lines 2190–2247)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 2192 | four times | IC authority vs roughness | `RES:1936` "~4x the authority of roughness"; 0.392 / 0.098 = 4.00 | EXACT |
| 2194 | $0.392$ | dMAE per unit fractional change, IC | `RES:1945` "**0.392**" | EXACT |
| 2195 | $a \in [0.8, 1.1]$ | well-sampled range | `RES:1945` "(over the well-sampled [0.8, 1.1])" | EXACT |
| 2195 | $0.098$ | same for uniform roughness | `RES:1946` "**0.098**" | EXACT |
| 2195 | $20\%$ | like-for-like perturbation | `RES:1946–1947` "Like for like at a 20% perturbation" | EXACT |
| 2195 | $0.075$\,m | IC at 20% | `RES:1947` "0.075 m versus 0.020 m" | EXACT |
| 2196 | $0.020$\,m | roughness at 20% | `RES:1947` | EXACT |
| 2196 | $20\%$ | water-storage error | `RES:1948` | EXACT |
| 2197 | 29-hour | spin-up | `RES:1948` "after a 29-hour spin-up under radar rainfall" | EXACT |
| 2198 | $30\%$ | of the published table | `RES:1949–1950` "Manning at 30% of the published table" | EXACT |
| 2199 | table's own value for open land | 0.040 | `RES:2106` | EXACT |
| 2204 | about $1\%$ | straightness | `RES:1952` | EXACT |
| 2205 | $a = 0.8$ and $1.1$ | range | `RES:1952` | EXACT |
| 2207 | $a = 0.6$ | extended scan point | `RES:2256–2257`; `logs/o57/o57_ic0.6.log:81`; `logs/o60/o60_ic0.6.log:81` | EXACT |
| 2207 | $40\%$ reduction | antecedent water | 1 − 0.6 = 0.40; `PS:68` "antecedent water −40%" | EXACT |
| 2208 | $0.5818$\,m | MAE at a = 0.6 | `logs/o57/o57_ic0.6.log:81` "J 5.969400e+02, peak-WSE MAE 0.5818 m, model dry at 0 of 46 marks"; `logs/o60/o60_ic0.6.log:81` (identical, independent allocation) | EXACT |
| 2213 | hour 29 | window start | `RES:1957` | EXACT |
| 2217 | $3$ unknowns | per cell in the initial state | `RES:1937–1938`, `RES:1960` "~8.8M unknowns"; 8.8M / 2.93M = 3.0 | DERIVED |
| 2218 | 2.93M | cells | `RES:1669` "2.93M cells"; `RES:278` | EXACT |
| 2218 | 46 | peak observations | `RES:1960` "against 46 peak observations" | EXACT |
| 2220 | three more orders of magnitude | of freedom | not stated anywhere; grepped "orders of magnitude", "8.8M", "2.93M" in `plans/`. 8.8M vs 15 classes is ~5.8 orders; 8.8M vs 46 obs is ~5.3 orders; 2.93M vs the paper's "fifteen classes" is ~5.3. No comparator gives 3 | **NOT FOUND** (and no reading of the numbers yields 3) |
| 2234 | 37 | marks the model cannot drain | `RES:1965` "the 37 marks the model cannot drain"; `RES:1383` | EXACT |
| 2235 | $20\%$ | water-balance error | `RES:1948` | EXACT |
| 2236 | hour 29 | | `RES:1957` | EXACT |
| 2238 | one to two months | spin-up needed | `TDL:178` "long-term simulation" (one to two months of spin-up before Harvey)" | EXACT |
| 2239 | 29 hours | current spin-up | `RES:1948` | EXACT |
| 2239 | $20\%$ | ordinary hydrologic scale | `RES:1948`; `PS:272` | EXACT |
| 2242 | kiang2018streamflow | streamflow uncertainty of that order | citation | N/A-citation |
| 2243 | xu2025harvey | precipitation/mesh dominance | citation | N/A-citation |
| 2245 | $20\%$ | illustrative perturbation | `RES:1947–1948`; `PS:272` | EXACT |

---

# Summary of problems

## NOT FOUND

1. **Line 2186, tab:ic caption — "No mark is dry at any point."** Dry counts are
   documented only for `a = 1.0` (0/46, `campaigns/o54_ic_authority.sh:91`),
   `a = 0.7` and `a = 0.6` (0/46, `logs/o57/o57_ic0.6.log:81`,
   `logs/o60/o60_ic0.6.log:81`, `RES:2286–2287`). The four scanned points that
   the table actually shows — `a = 0.8, 0.9, 1.1, 1.2` — have no recorded dry
   count anywhere: `RES:1940–1943` has no dry column, and no o54 raw log exists
   in the repo. Grepped every J and MAE value of those four rows plus `"dry at"`
   across `plans/` and `logs/`.

2. **Line 2220 — "three more orders of magnitude of freedom."** No source, and
   no arithmetic over the available counts (8.8M state unknowns, 2.93M cells,
   46 observations, 15 classes) produces three orders; the natural comparisons
   give five to six. Grepped "orders of magnitude", "8.8M", "2.93M" in `plans/`.

## CONTRADICTED

3. **Line 1433, tab:alpha caption — "one mark stays dry throughout."** The source
   sentence `RES:1592` reads "The limit is NOT drying (1/46 dry throughout)" but
   is scoped to the o44/o45 alphas 0.3/0.45/0.6. The table it captions reports
   0/46 at α = 0.80 and α = 1.00 (`RES:2265–2266`, `RES:1582`), so "throughout"
   is contradicted by the table's own last two rows.

## Provenance caveat (not an error, but worth recording)

4. **tab:alpha, tab:halves and tab:ic rest entirely on the results record, not on
   raw logs.** `find` over the repo shows no artifacts for o44/o45 (alpha scan),
   o47 (halves), o49 (α = 0.70/0.80), or o54 (IC scan) — only
   `plans/campaigns/o49_alpha_bar.sh` and `plans/campaigns/o54_ic_authority.sh`.
   Every value in those three tables traces to `RESULTS-gpu-implicit.md` alone.
   By contrast the tab:ladder rows, the 0.5818 point and the spectrum counts do
   have raw logs (`logs/o59/`, `logs/o62/`, `logs/o57/`, `logs/o60/`,
   `logs/o61/`).

## Rounding worth a second look (all defensible, listed for completeness)

- Line 1576 "three and a half times" against the source's "~3.6x" (`RES:2075–2076`).
- Line 1565 shrub `0.035` from `0.0345` (`RES:2444`) — rounds up at the boundary.
- Line 1466 "$85\%$ or more" against `RES:1597`'s "~85-90%" — the paper's phrasing
  covers the band, but the later Sec 6.3/Sec 7 "$85\%$" (lines 1595, 2152) is the
  strict 100 − 15 complement and is a different derivation from the same figure.
