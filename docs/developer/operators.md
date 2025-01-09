# Composite Operator Structure

In order to implement the physics necessary for assessing risks surrounding
compound flooding, RDycore allows the construction of _operators_ that solve
sets of equations using _sub-operators_ as building blocks. Each of these
sub-operators is implemented by either a `CeedOperator` or a set of functions
implementing a PETSc right-hand-side (RHS) function.

## What is an operator?

For the purposes of our discussion, an operator transforms a solution vector at
a time $t$ to its time derivative, using the language of ordinary differential
equations. A linear operator $\mathcal{L}$ represents a linear transformation,
which can be thought of like a matrix:

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathcal{L}(t)\mathbf{u}
$$

Meanwhile, a nonlinear operator $\mathcal{N}$ behaves like a function:

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathcal{N}(t, \mathbf{u})
$$

For example, in a conservative system of equations solved by the finite volume
method

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot\vec{\mathbf{F}}(\mathbf{u}) = \mathbf{S}(t, \mathbf{u}),
$$

the relevant nonlinear operator can be written

$$
\mathcal{N}(t, \mathbf{u}) = -\nabla\cdot\vec{\mathbf{F}}(\mathbf{u}) + \mathbf{S}(t, \mathbf{u}).
$$

Each sub-operator within an operator assumes responsibility for part of this
calculation. Let's look at some examples.

### Shallow Water Equations (SWE) operator

This operator consists of the following sub-operators, each of which operates
on all 3 components of a solution vector $\mathbf{u} = [h, hu, hv]^T$:

1. an "interior flux operator" that computes fluxes on edges separating pairs of
   cells on the interior of the domain
2. a "boundary flux operator" for each boundary on the domain that computes
   fluxes on boundary edges using adjoining interior cells and values assigned 
   to each boundary
3. a "source operator" that computes source terms on the interior of the domain

### Sediment Transport operator



## CEED implementation


