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
  constants:
    h0: 1
    n0: 1
    u0: 1
    v0: 1
    z0: 1
  solutions: # functions of x, y, t for the shallow water equations (non-normalized units)
    # water height
    h:    h0 * (1 + sin(x)*sin(y)) * exp(t)
    dhdx: h0 * exp(t) * sin(y) * cos(x)
    dhdy: h0 * exp(t) * sin(x) * cos(y)
    dhdt: h0 * (1 + sin(x)*sin(y)) * exp(t)

    # x velocity
    u:     u0 * cos(x) * sin(y) * exp(t)
    dudx: -u0 * sin(x) * sin(y) * exp(t)
    dudy:  u0 * cos(x) * cos(y) * exp(t)
    dudt:  u0 * cos(x) * sin(y) * exp(t)

    # y velocity
    v:     v0 * sin(x) * cos(y) * exp(t)
    dvdx:  v0 * cos(x) * cos(y) * exp(t)
    dvdy: -v0 * sin(x) * sin(y) * exp(t)
    dvdt:  v0 * sin(x) * cos(y) * exp(t)

    # (x and y momenta can be evaluated from h(x,y,t) * {u,v}(x,y,t))

    # elevation as z(x, y)
    z:     z0 * sin(x) * sin(y)
    dzdx:  z0 * cos(x) * sin(y)
    dzdy:  z0 * sin(x) * cos(y)

    # source-related term n(x,y)
    n:     n0 * (1 + sin(x) * sin(y))
```

The `mms` section defines the forms of the manufactured solutions for the
model equations corresponding to parameters set in the `physics` section.

The `constants` subsection defineѕ a set of named constants that can be used
in the analytic forms for the manufactured solutions. In the above example, the
solutions reference the constants `h0`, `u0`, `v0`, `z0`, and `n0`, which
are defined as shown.

The `solutions` subsection defines a set of manufactured solutions in terms of
functions of the coordinates `x` and `y` and the time `t`. These solutions are
parsed and compiled at runtime so they can be evaluated as needed by the model.
This means you can define a new manufactured solution in every MMS driver input
file, without developing code and rebuilding RDycore.
