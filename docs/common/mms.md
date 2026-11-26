# The Method of Manufactured Solutions (MMS) Verification Driver

## Input

The MMS driver accepts input in YAML form, like the main RDycore driver. However,
the MMS driver's input has a ѕlightly different form. Like the [main driver input](input.md),
it's organized into several sections. Many of these sections are identical to
those in the main driver's input:

* **Model equations and discretizations**
    * [physics](input.md#physics)
    * [numerics](input.md#numerics)
    * [grid](input.md#grid)
    * [regions](input.md#regions)
    * [boundaries](input.md#boundaries)
    * [time](input.md#time)
* **Simulation diagnostics, output, and restarts**
    * [logging](input.md#logging)
    * [output](input.md#output)

However, the other sections, which define material properties, initial/boundary
conditions, and sources, are not present in the MMS input. This is because the
method manufactured solutions requires analytic forms for these terms to produce
a convergent manufactured solution. So these sections are replaced by a single
`mms` section that defines these analytic forms.

### `mms` section

```yaml
mms:
  constants: # any single capital letter can be used
    H: 0.005  # water height scale factor
    T: 20.0   # time scale
    U: 0.025  # x-velocity scale factor
    V: 0.025  # y-velocity scale factor
    N: 0.01   # manning coefficient scale factor
    Z: 0.0025 # elevation scale factor

    K: 0.6283185307179586 # wave number in x and y (pi/5)
  swe: # functions of x, y, t (non-normalized units)
    # water height
    h:    H * (1 + sin(K*x)*sin(K*y)) * exp(t/T)
    dhdx: H * K * sin(K*y) * cos(K*x) * exp(t/T)
    dhdy: H * K * sin(K*x) * cos(K*y) * exp(t/T)
    dhdt: H / T * (1 + sin(K*x)*sin(K*y)) * exp(t/T)

    # x velocity
    u:     U * cos(K*x) * sin(K*y) * exp(t/T)
    dudx: -U * K * sin(K*x) * sin(K*y) * exp(t/T)
    dudy:  U * K * cos(K*x) * cos(K*y) * exp(t/T)
    dudt:  U / T * cos(K*x) * sin(K*y) * exp(t/T)

    # y velocity
    v:     V * sin(K*x) * cos(K*y) * exp(t/T)
    dvdx:  K * V * cos(K*x) * cos(K*y) * exp(t/T)
    dvdy: -K * V * sin(K*x) * sin(K*y) * exp(t/T)
    dvdt:  V / T * sin(K*x) * cos(K*y) * exp(t/T)

    # elevation as z(x, y)
    z:     Z * sin(K*x) * sin(K*y)
    dzdx:  Z * K * cos(K*x) * sin(K*y)
    dzdy:  Z * K * sin(K*x) * cos(K*y)

    # Manning coefficient n(x,y)
    n:     N * (1 + sin(K*x) * sin(K*y))

  # Manufactured water temperature, required when physics.heat is enabled
  temperature:
    T:    A * (1 + sin(K*x)*sin(K*y)) * exp(t/T)
    dTdx: A * K * sin(K*y) * cos(K*x) * exp(t/T)
    dTdy: A * K * sin(K*x) * cos(K*y) * exp(t/T)
    dTdt: A / T * (1 + sin(K*x)*sin(K*y)) * exp(t/T)

  # Convergence study parameters (optional)
  convergence:
    num_refinements: 3
    base_refinement: 1
    timestep_refinement_exponent: 1
    expected_rates:
      h:
        L1: 1
        L2: 1
        Linf: 0.48
      hu:
        L1: 0.73
        L2: 0.78
        Linf: 0.62
      hv:
        L1: 0.73
        L2: 0.78
        Linf: 0.62
      hT:
        L1: 0.93
        L2: 0.92
        Linf: 0.92
      T:
        L1: 0.90
        L2: 0.88
        Linf: 0.72
```

The `mms` section defines the forms of the manufactured solutions for the
model equations corresponding to parameters set in the `physics` section.

The `constants` subsection defineѕ a set of named constants that can be used
in the analytic forms for the manufactured solutions. Any single capital roman
letter (`A` through `Z`) can be used as a constant. In the above example, the
solutions reference the constants `H`, `T`, `U`, `V`, `N`, `Z`, and `K`, which
are defined as shown.

The `swe` subsection defines a set of manufactured solutions to the 2D shallow
water equations (SWE) in terms of a water height `h` with a flow velocity
`(u, v)`. Each of the components `h, u, v` are represented by a function of the
coordinates `x` and `y` and the time `t`. Other model parameters (`z`, the
elevation function, and `n`, the Manning coefficient) are functions of `x` and
`y` only.

The `temperature` subsection is required when `physics.heat` is enabled. It gives
the manufactured water temperature `T` and its partial derivatives. Note that
RDycore's prognostic heat variable is the conservative quantity `hT`, not `T`
itself; the driver forms `h*T` when it initializes and evaluates the exact
solution, and reports error norms for both (see
[Heat error norms](#heat-error-norms) below).

Manufactured expressions are limited to 128 characters, so a long derivative may
have to be written without spaces.

These analytic forms are parsed and compiled at runtime so they can be evaluated
as needed by the model. This means you can define a new manufactured solution in
every MMS driver input file, without developing code and rebuilding RDycore.

### Convergence studies

The optional `convergence` subsection contains the following parameters for
performing convergence studies that determine whether the MMS problem has been
solved successfully for each solution component:

* `num_refinements`: the number of times the domain is refined uniformly from
  the base resolution to test the rate of convergence of the solution error.
  This parameter is required.
* `base_refinement`: this optional parameter specifies the number of times the
  mesh should be refined to establish the coarsest resolution to be used in the
  convergence study. For example, a `base_refinement` of 2 indicates that a
  mesh loaded from a file should be refined twice before performing a
  convergence study.
* `timestep_refinement_exponent`: an optional exponent `p` giving the timestep
  used at refinement level `r` as `dt_0 / 2^(p*r)`, where `dt_0` is the timestep
  configured in the `time` section. `stop_n` is scaled by the reciprocal so that
  every level ends at the same physical time. The default of `0` holds the
  timestep fixed across levels, which can let temporal and operator-splitting
  error dominate the finest meshes; `1` refines `dt` with `dx`, and `2` refines
  it with `dx^2`.

    Note that with `p = 1` and first-order discretizations in both space and
    time, the error behaves as `E ~ Cx dx + Ct dt = (Cx + K Ct) dx`, so the
    reported rate is a **joint space-time rate**, not an isolated spatial one.
    Isolating the spatial order requires a timestep strategy under which the
    temporal error is demonstrably negligible or asymptotically smaller --
    `numerics.temporal: rk4`, a fixed `dt` shown to be small enough, or `p = 2`
    (correct, but the work scales as `dx^-4`).

* `expected_rates`: a sub-subsection with `L1`, `L2`, and `Linf`
  entries for each relevant solution component name giving the expected rates of
  convergence for the appropriate error norms. Each of the component names and
  expected rates are optional, so you can specify only those you want to use
  as pass/fail criteria.

**NOTE: When the MMS driver performs a convergence study, it writes no output.
If you need to write a mesh or solution data, you can always use an input file
without the `convergence` section to compute error norms for a single spatial
resolution, writing output as needed.**

### Heat error norms

When `physics.heat` is enabled the driver reports two rows for the heat solution:

```text
  hT : L1 = ..., L2 = ..., Linf = ...
   T : L1 = ..., L2 = ..., Linf = ...
```

* `hT` is the error in the conservative variable the solver actually advances.
* `T` is the error in the derived temperature `hT / h`, guarded by
  `physics.flow.tiny_h` exactly as the operators guard it, and compared against
  the manufactured `T` evaluated at the same cell centroids. This is the
  physically meaningful quantity, and it can combine or partially cancel errors
  in `h` and `hT`, so it generally converges at a slightly different rate.

Both carry independent `expected_rates` entries (`hT` and `T`). They coincide
only when `h` is identically 1.

### Reference-solution self-convergence

Two command-line options support measuring temporal (and operator-splitting)
order on a fixed mesh:

* `-mms_save_final_state <file>` writes the final prognostic state to a PETSc
  binary file.
* `-mms_reference_solution <file>` loads such a file and reports componentwise
  difference norms against the current solution, prefixed with `ref`.

These exist because refining `dt` on a fixed mesh does **not** drive the error
against the exact solution to zero once transport is active: the solution
converges to the semi-discrete (method-of-lines) solution, whose distance from
the exact solution is the spatial truncation error. A study against the exact
solution therefore plateaus on that floor and reports a rate near zero.
Differencing two runs that share a spatial discretization cancels the floor and
leaves the temporal and splitting error alone. Choose the reference timestep at
or below `dt_min / 16` -- at first order the reference's own error biases the
finest measured point by roughly `dt_ref / dt_min` -- and validate it by halving
it once and confirming the reported errors barely move.

The file stores the global vector in DMPlex's distributed ordering, so a
reference is only comparable to a run on the same mesh with the same number of
MPI ranks.

### Scope of the heat MMS path

The MMS driver drives the heat source solve through the `direct_source` branch,
which **replaces** the nonlinear atmospheric parameterization `Q_net(T)` rather
than correcting it. The heat MMS cases therefore verify passive transport of
`hT`, the manufactured source quadrature, and the Lie composition of the two
solves -- but they do **not** verify `Q_net(T)` or its analytic Jacobian.

A consequence worth stating: in this path the heat residual `f = udot - S/(rho c)`
has no state dependence, since `S` is a per-cell array that does not vary with
the stage time. Every consistent one-step method therefore produces the same
update, and the heat TS type (`-heat_ts_type beuler` or `cn`) affects only which
manufactured quadrature the driver installs -- the right endpoint for backward
Euler, the endpoint average for Crank-Nicolson. It does not affect the order of
the source step.
