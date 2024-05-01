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

    # Convergence study parameters (optional)
    convergence:
      num_refinements: 3
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

These analytic forms are parsed and compiled at runtime so they can be evaluated
as needed by the model. This means you can define a new manufactured solution in
every MMS driver input file, without developing code and rebuilding RDycore.

### Convergence studies

The `convergence` sub-subsection is optional and contains the following
parameters for performing convergence studies that determine whether the MMS
problem has been solved successfully for each solution component:

* `num_refinements`: the number of times the domain (and timestep) are refined
  uniformly to test the rate of convergence of the solution error
* `expected_rates`: a sub-subsection with `L1`, `L2`, and `Linf`
  entries giving the expected rates of convergence for the appropriate error
  norms.
