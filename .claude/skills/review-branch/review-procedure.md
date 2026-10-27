# RDycore review procedure

Ordered by how many of the eight bug fixes in the AMR series (2026-10) each step
would have caught:

| bug | what would have exposed it | step |
|---|---|---|
| argument checks compiled out of optimized builds | build type | 1 |
| boundary conditions not rebuilt with the boundaries | rank count + repartition | 1 |
| CEED flux offsets indexed by the owned permutation | operator backend | 1, 3, 4 |
| `u_local` read before it is filled, under `-refine_at_start` | option combination | 1 |
| free surface vs depth carried across an adapt | a physical invariant asserted exactly | 2 |
| a bed read from a file across an adapt | input source | 1 |
| refinement cap on a graded mesh | mesh type | 1 |
| coarse-DM chain closing on itself | run duration | 1 |

Step 1 six, step 2 the subtlest, steps 3 and 4 one each, **step 5 zero** —
measured, by giving a thorough reviewer the branch with a known fix removed and
watching it miss, because the defect was in a file the diff never touches. Step 5
still earns its place: run on that same branch it found three defects nothing
else did. Re-derive this table when the next few bugs are found; if step 1 stops
leading, reorder.

---

## 1. Configuration-axis coverage

Most bugs here were configuration-shaped — the code was right on the axis it was
developed on. **Use the `axis-coverage` skill**, or do it by hand: for each axis
below, ask whether the change interacts with it and whether a test exists on the
far side.

Axes: operator backend (`basic`/`ceed`), timing mode (`-preload`), rank count
(1/2/3), build type (debug/optimized), PETSc index size (int32/int64), mesh type
(uniform tri / graded / quad / mixed / `.exo` DEM), input source (geometry vs
`grid.cell_elevation` file), option combinations, run duration.

The suite crosses backend, timing mode and rank count. It does **not** cross
build type, index size, mesh type, input source or duration — and three of the
eight bugs lived in exactly those.

Report an axis as a **coverage gap**, not a finding, unless you can name a
concrete failure on it.

## 2. Exactness-class invariant assertions

For each behavioural change, ask what must remain **exactly** true, and whether a
gate asserts it with its class named:

| class | meaning |
|---|---|
| bit-identical | the answer must not move at all |
| exact degeneration | provably a no-op in some configuration |
| machine precision, with the mechanism | only summation order changes — say why |
| physics rate | a convergence rate, or error against an exact solution |

A change with no named class and no gate is a finding (Style at least). A gate
whose class is "machine precision" without a stated mechanism is the same.

**Beware the invariant that is conserved by the bug.** `sum(h*area)` is invariant
under a permutation of values among cells, because each interior face still
contributes `+F` to one cell and `−F` to another however they are misdirected.
That is why `-refine_check_volume` never saw the CEED flux bug. A partition- and
order-independent checksum that *does* see it uses weighted moments —
`sum(h²·area)`, `sum(h·x·area)`, `sum(h·y·area)`, `max h`. Serial against 3 ranks
agrees on those to 0.00e+00, 0.00e+00 and one ulp respectively, so they can be
held at ~1e-14. Script: `.claude/skills/backend-diff/moments.py`.

## 3. Backend-differential run

Two independent implementations of the same mathematics must agree. **Use the
`backend-diff` skill.** Machine precision when correct (relative 3.3e-16 on the
channel case); on a branch with the CEED flux bug the two backends do not even
produce the same mesh.

**Its blind spot:** it only sees a defect present in exactly one of the two
implementations. Seven of the eight bugs are in shared code, where both backends
compute the same wrong answer. Run it; do not rely on it.

## 4. Index-map audit

**Use the `index-map-audit` skill.** The cheapest step and the only reading-based
one that catches an invariant break in untouched code. On its first run it found
the target bug plus two mismatches pre-existing on `main`, one of them a read
past the end of a Vec.

Run it when the change touches the mesh, ownership, partitioning, an operator, or
any `*_to_*` array. Skip it for a documentation-only or build-only change and say
so.

## 5. Read the diff

Review correctness, error handling, documentation and style. RDycore has no
`AGENTS.md` and no style guide, so **say so as a coverage limitation** and
confine style findings to internal inconsistency and to user-facing text that
contradicts the code.

- Treat `PetscCall()`, `PetscCheck()` and `SETERRQ` as terminal on error. Report
  resource errors on normal paths, not missing cleanup on fatal paths.
- Flag any PETSc call not wrapped in `PetscCall`/`PetscCallMPI`/`PetscCallAbort`.
- `clang-format` is pinned to **14** (`CMakeLists.txt`). If only another version
  is installed, do not run it — report formatting as unverified.
- Never review `.out` contents. These tests assert inside CMake scripts, so there
  are usually none.

### Severity

| severity | required evidence |
|---|---|
| CRITICAL / HIGH / MEDIUM | a concrete trigger (an input, an option, a rank count or partition condition) **and** a concrete impact: wrong result, crash, hang, leak, or a named slowdown |
| Style | for a convention point, name the inconsistency; for factually wrong user-facing text, quote it and state what the code does |
| LOW | awkward-but-accurate wording, lost convenience, one cheap rerun. Count them; list only on request |

A design alternative is not a finding. Factually wrong user-facing text is never
LOW. No findings is a valid result.

**Verify before reporting.** Read the callee, the macro, the allocation — at the
reviewed revision, not the working tree, which may be on another branch. Discard
what verification disproves and say what evidence was missing where it could not
be resolved.

## 6. Verify the gate claims

**Use the `gate-claims` skill.** For every gate a commit message names, revert
that commit and confirm exactly the named gates fail.

**Build the revert with `git revert`, never by pattern-matching.** A partial
revert produces a false finding that the claim is overstated, which is worse than
not checking because it reads like diligence. This has happened: a hand-written
pattern revert missed one call site in one file and then an entire second file,
and the wrong conclusion was drawn from it twice.

---

## Report

- Findings first, ordered CRITICAL → HIGH → MEDIUM → Style. For each: the
  problem, the evidence, `file:line` for every occurrence, and a suggested fix.
  Where the fix is small and unambiguous, give a fenced `diff`; label it untested
  unless you built and ran it, and say so separately from the finding's own
  verification.
- Then **coverage**: which steps were run, which were skipped and why, which
  files and lines were read, what was not covered (binaries, generated meshes),
  and every verification limitation.
- Then **axis gaps** from step 1, listed as gaps rather than findings.
- End with `(N LOW findings suppressed; ask to show them.)` when `N > 0`.
- No praise, no summary of the branch, no design commentary. If there is nothing
  at or above Style, say exactly that.
