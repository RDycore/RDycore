# Shallow Water Equations

The two-dimensional shallow water equations can be written in the conservative
form

$$
\frac{\partial\mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = \mathbf{S}\label{1}
$$

where

\begin{align}
  \mathbf{U}
  =
  \begin{bmatrix}
  h \\[.5em]
  hu \\[.5em]
  hv
  \end{bmatrix},
\end{align}

\begin{align}
  \mathbf{E}
  =
  \begin{bmatrix}
  hu \\[.5em]
  hu^2 + \frac{1}{2}gh^2 \\[.5em]
  huv
  \end{bmatrix},
\end{align}

\begin{align}
  \mathbf{G}
  =
  \begin{bmatrix}
  hv \\[.5em]
  huv \\[.5em]
  hv^2 + \frac{1}{2}gh^2
  \end{bmatrix},
\end{align}

\begin{align}
  \mathbf{S}
  =
  \begin{bmatrix}
  Q \\[.5em]
  -gh\frac{\partial z}{\partial x} - C_D u \sqrt{u^2 + v^2}\\[.5em]
  -gh\frac{\partial z}{\partial y} - C_D v \sqrt{u^2 + v^2}
  \end{bmatrix}.
\end{align}

In the above expressions,

* $h$ is the flow depth
* $u$ is the vertically-averaged velocity in the $x$ direction
* $v$ is the vertically-averaged velocity in the $y$ direction
* $z$ is the bed elevation (generally a function of $x$ and $y$)
* $C_D = g n^2/h^\frac{1}{3}$ is the drag coefficient
* $n$ is the Manning's coefficient

It is sometimes convenient to interpret the terms involving $\mathbf{E}$ and
$\mathbf{G}$ as the (two-dimensional) divergence of a multi-component spatial
vector called the _flux function_ $\mathbf{\vec{F}}$.

$$
\frac{\partial\mathbf{U}}{\partial t} + \vec{\nabla}\cdot\mathbf{\vec{F}}(\mathbf{U}) = \mathbf{S}(\mathbf{U})\label{2}
$$

where $\mathbf{\vec{F}} = (\mathbf{F}_x, \mathbf{F}_y) = (\mathbf{E}, \mathbf{G})$.

We have written $\mathbf{\vec{F}}$ and $\mathbf{S}$ in $(\ref{2})$ in a way that
emphasizes that these functions depend on the solution vector $\mathbf{U}$.

## Spatial Discretization

We can rewrite the shallow water equations in a form more convenient for
numerical treatment by defining a computational domain $\Omega$ bounded by a
piecewise linear closed curve $\Gamma = \partial\Omega$.

We create a discrete representation by partitioning $\Omega$ into disjoint
cells, with $\Omega_i$ representing cell $i$. The boundary of cell $i$, written
$\partial\Omega_i$, is the set of faces separating it from its neighboring
cells. Using this notation, we obtain a discrete set of equations for the
solution in cell $i$ by integrating $(\ref{1})$ over $\Omega_i$ and using
Green's theorem:

\begin{eqnarray}
\frac{\partial}{\partial t} \int_{\Omega_i} \mathbf{U} d\Omega_i +
\int_{\Omega_i} \left[ \frac{\partial\mathbf{E}}{\partial x} +
\frac{\partial\mathbf{G}}{\partial y} \right] d\Omega_i &=&
\int_{\Omega_i} \mathbf{S} d\Omega_i \nonumber\\
\frac{\partial}{\partial t} \int_{\Omega_i} \mathbf{U} d\Omega_i +
\oint_{\partial\Omega_i} \left( \mathbf{E}~dy - \mathbf{G}~dx \right) &=&
\int_{\Omega_i} \mathbf{S} d\Omega_i \label{3}
\end{eqnarray}

This equation can be used to approximate discontinuous flows, because all
quantities appear under integrals. By contrast, $(\ref{1})$ cannot be used
where derivatives of $\mathbf{U}$ don't exist.

We can interpret the line integral in ($\ref{3}$) in terms of the flux
$\mathbf{\vec{F}} = (\mathbf{F}_x, \mathbf{F}_y)$ between a cell $i$ and its
neighboring cells.

$$
 \frac{\partial}{\partial t} \int_{\Omega_i} \mathbf{U} d\Omega_i +
\oint_{\partial\Omega_i} \mathbf{\vec{F}} \cdot \vec{n}~dl =
\int_{\Omega_i} \mathbf{S} d\Omega_i \label{4}
$$

Here, we have defined a unit normal vector $\vec{n} = (n_x, n_y)$
pointing outward along the cell boundary $\partial\Omega_i$. $(\ref{4})$ is a
"surface integral" with a differential arc length $dl$ integrated over the
boundary of cell $i$. One obtains this surface integral by integrating $(\ref{2})$
over the domain $\Omega$ and applying the (two-dimensional) divergence theorem
to the flux term.

To compare these two integral forms, consider the line integral in $(\ref{3})$
on a quadrilateral cell along a counterclockwise path connecting its vertices:

\begin{eqnarray}
I_1 =  &\oint_{\partial\Omega_i}& \left( \mathbf{E}~dy - \mathbf{G}~dx \right) \\
   = - &\int_{x_1}^{x_2}&\mathbf{G}|_{y_1}~dx ~~~~\mathrm{(bottom)} \\
     + &\int_{y_1}^{y_2}&\mathbf{E}|_{x_2}~dy ~~~~\mathrm{(right)} \\
     - &\int_{x_2}^{x_1}&\mathbf{G}|_{y_2}~dx ~~~~\mathrm{(top)} \\
     + &\int_{y_2}^{y_1}&\mathbf{E}|_{x_1}~dy ~~~~\mathrm{(left)}
\end{eqnarray}

Compare this to the corresponding term in $(\ref{4})$, a surface integral in
which the areas (lengths) of the faces (edges) are always positive:

\begin{eqnarray}
I_2 = &\iint_{\partial\Omega_i}& \mathbf{\vec{F}} \cdot \vec{n}~dl_i \\
     - &\int_{x_1}^{x_2}&\mathbf{F}_y|_{y_1}~dx ~~~~\mathrm{(bottom)} \\
     + &\int_{y_1}^{y_2}&\mathbf{F}_x|_{x_2}~dy ~~~~\mathrm{(right)} \\
     + &\int_{x_1}^{x_2}&\mathbf{F}_y|_{y_2}~dx ~~~~\mathrm{(top)} \\
     - &\int_{y_1}^{y_2}&\mathbf{F}_x|_{x_1}~dy ~~~~\mathrm{(left)}
\end{eqnarray}

Evidently $I_1 = I_2$ if $\mathbf{F}_x = \mathbf{E}$ and $\mathbf{F}_y = \mathbf{G}.$

In the rest of this section, we use the flux form $(\ref{4})$ of the shallow
water equations.

### Finite volume formulation

We can obtain a finite volume method for these equations by defining
_horizontally-averaged_ quantities for flow depth and velocities:

\begin{eqnarray}
h_i &=& \frac{1}{A_i}\int_{\Omega_i} h d\Omega_i \\
u_i &=& \frac{1}{A_i}\int_{\Omega_i} u d\Omega_i \\
v_i &=& \frac{1}{A_i}\int_{\Omega_i} v d\Omega_i
\end{eqnarray}

where $A_i = \int_{\Omega_i}d\Omega_i$ is the area enclosed within cell $i$. We
also introduce the _horizontally-averaged solution vector_

$$
\mathbf{U}_i = (h_i, h_i u_i, h_i v_i)^T
$$

and the _horizontally-averaged source vector_

$$
\mathbf{S}_i = \frac{1}{A_i}\int_{\Omega_i} \mathbf{S} d\Omega_i.
$$

Finally, we define the _face-averaged normal flux vector_ between cell $i$ and
an adjoining cell $j$:

$$
\mathbf{F}_{ij} = \frac{1}{l_{ij}}\int_{\partial\Omega_i\bigcap\partial\Omega_j}\mathbf{\vec{F}}\cdot\vec{n}~dl \label{5}
$$

where $l_{ij}$ is the length of the face connecting cells $i$ and $j$.

With these definitions, the shallow water equations in cell $i$ are

$$
\frac{\partial\mathbf{U}_i}{\partial t} + \sum_j\mathbf{F}_{ij} l_{ij} = \mathbf{S}_i, \label{6}
$$

where the index $j$ in each term of the sum refers to a neighboring cell of cell $i$.

### Boundary conditions

To incorporate boundary conditions, we partition the domain boundary $\Gamma$
into disjoint line segments, each of which represents a _boundary face_
$\Gamma_i$. Every cell $j$ that touches the boundary $\Gamma$ has at least one
boundary face so that 

$$
\partial\Omega_j \bigcup \Gamma \neq \varnothing.
$$

Such a cell is a _boundary cell_. The boundary $\Gamma$ consists entirely of
faces of boundary cells.

In dealing with boundary conditions, we must distinguish between the faces a
boundary cell $i$ _does_ and _does not_ share with the boundary $\Gamma$:

$$
\frac{\partial\mathbf{U}_i}{\partial t} +
\sum_{j: \partial\Omega_j\subset\Gamma}\mathbf{F}_{ij}^{\Gamma} l_{ij} +
\sum_{j: \partial\Omega_j\not\subset\Gamma}\mathbf{F}_{ij} l_{ij}
= \mathbf{S}_i. \label{7}
$$

To enforce boundary conditions, we must compute the effective boundary fluxes
$\mathbf{F}_{ij}^{\Gamma}$ that appear in the first sum in $(\ref{7})$. These
boundary fluxes have specific forms depending on their respective boundary
conditions.

### Source terms

The term for the vector-valued source for the water's momentum, which we
designate as $\vec{S}_{\vec{p}}$ is computed from the following integral:

\begin{equation}
\int_{d\Omega} \mathbf{\vec{S}}_\vec{p} d\Omega = \int_{\Omega} \left( -gh\nabla z - C_D \vec{u} \sqrt{u^2 + v^2} \right) d\Omega
\end{equation}

Here, $\nabla z = (\partial z/\partial x, \partial z/\partial y)$ is the
two-dimensional gradient of the bed elevation function $z$ and $\vec{u} = (u, v)$
is the flow velocity vector.

The first of these terms, the _bed elevation slope_ term, represents the force
of gravity on the water and can be approximated as

\begin{equation}
\int_{d\Omega_i} -gh\nabla z d\Omega_i \approx -gh\left(\overline{\frac{\partial z}{\partial x}}, \overline{\frac{\partial z}{\partial y}}\right) V_i
\end{equation}

where $V_i$ is the volume of cell $i$. For a triangular grid cell,

\begin{eqnarray}
\overline{\frac{\partial z}{\partial x}} &=& \frac{(y_2 - y_0)(z_1 - z_0) - (y_1 - y_0)(z_2 - z_0)}{(y_2 - y_0)(x_1 - x_0) - (y_1 - y_0)(x_2 - x_0)} \\
\overline{\frac{\partial z}{\partial y}} &=& \frac{(x_2 - x_0)(z_1 - z_0) - (x_1 - x_0)(z_2 - z_0)}{(x_2 - y_0)(x_1 - x_0) - (x_1 - y_0)(y_2 - y_0)}.
\end{eqnarray}

The second term, the _roughness term_, is associated with bed friction:

\begin{equation}
\int_{\Omega_i} C_D \vec{u} \sqrt{u^2 + v^2} d\Omega_i \approx C_D \vec{u} \sqrt{u^2 + v^2} V_i
\end{equation}

### Evaluating normal fluxes

We have reduced the spatial discretization of our finite volume method to the
calculation of normal fluxes between neighboring cells and on boundary faces.
The normal flux function is

\begin{equation}
\mathbf{\vec{F}}\cdot\vec{n} = \mathbf{E} n_x + \mathbf{G} n_y.
\end{equation}

We can evaluate $\mathbf{F}_{ij} \approx \mathbf{\vec{F}}\cdot\vec{n}$, the
approximate normal flux at a face shared shared by _interior cells_ $i$ and $j$,
by solving the relevant Riemann problem using Roe's method. If we designate
$\phi$ as the angle between $\vec{n}$ and the $x$ axis and adopt $i$ and $j$
subscripts for quantities respectively in cells $i$ and $j$, we can approximate
the normal flux by the expression

\begin{equation}
\mathbf{F}_{ij} = \frac{1}{2} \left( \mathbf{\vec{F}}_i + \mathbf{\vec{F}}_j - \mathbf{\hat{R}} |\mathbf{\hat{\Lambda}| \mathbf{\Delta}\hat{V}} \right).
\end{equation}

Above,

\begin{align}
  \mathbf{R}
  =
  \begin{bmatrix}
  1 & 0 & 1  \\[.5em]
  \hat{u} - \hat{a}\cos\phi & -\sin\phi & \hat{u} + \hat{a}\cos\phi  \\[.5em]
  \hat{v} - \hat{a}\sin\phi &  \cos\phi & \hat{v} + \hat{a}\sin\phi
  \end{bmatrix}
\end{align}

\begin{align}
  \mathbf{\Delta\hat{V}}
  =
  \begin{bmatrix}
  \frac{1}{2} \left( \Delta h - \frac{\hat{h}\Delta u_\perp}{\hat{a}} \right) \\[.5em]
  \hat{h}u_\parallel \\[.5em]
  \frac{1}{2} \left( \Delta h + \frac{\hat{h}\Delta u_\perp}{\hat{a}} \right)
  \end{bmatrix}
\end{align}

\begin{align}
  |\mathbf{\hat{\Lambda}}|
  =
  \begin{bmatrix}
  | \hat{u}_\perp - \hat{a} |^* & 0 & 0  \\[.5em]
  0                                 & |\hat{u}_\perp| & 0 \\[.5em]  
  0                                 &                         & | \hat{u}_\perp + \hat{a} |^* 
  \end{bmatrix}
\end{align}

with

\begin{eqnarray}
  \hat{h} & = & \sqrt{h_i h_j} \\
  \hat{u} & = & \frac{ \sqrt{h_i} u_i + \sqrt{h_j} u_j}{ \sqrt{h_i} + \sqrt{h_j}} \\
  \hat{v} & = & \frac{ \sqrt{h_i} v_i + \sqrt{h_j} v_j}{ \sqrt{h_i} + \sqrt{h_j}} \\
  \hat{a} & = & \sqrt{\frac{g}{2} \left( h_i + h_j \right)}
\end{eqnarray}

and $\Delta f = f_j - f_i$ is the change in the quantity $f$ moving from cell
$i$ to $j$.

We have used asterisks in the expression for $|\mathbf{\hat{\Lambda}}|$ to
indicate that the eigenvalues 
$\hat{\lambda}_1 = \hat{u}_\perp - \hat{a}$ and
$\hat{\lambda}_3 = \hat{u}_\perp + \hat{a}$ 
must be adjusted, since Roe's method does not provide the correct flux for
critical flow.

\begin{eqnarray}
  |\hat{\lambda}|_1 &=& \frac{\hat{\lambda}^2_1}{\Delta \lambda} + \frac{\Delta \lambda}{4} \mbox{$~$ if $~ -\Delta \lambda/2 < \hat{\lambda}_1 < \Delta \lambda/2$} \\
  |\hat{\lambda}|_3 &=& \frac{\hat{\lambda}^2_2}{\Delta \lambda} + \frac{\Delta \lambda}{4} \mbox{$~$ if $~ -\Delta \lambda/2 < \hat{\lambda}_3 < \Delta \lambda/2$}
\end{eqnarray}
 
## Temporal Discretization

The above spatial discretization produces a "semi-discrete" system of equations
that can be integrated using various methods for solving systems of ordinary
differential equations. The following methods of integration are provided by
PETSc and supported for RDycore:

* [Forward Euler](https://petsc.org/release/manualpages/TS/TSEULER/)

## References

* [Bradford, S. F., & Sanders, B. F. (2002). Finite-volume model for shallow-water flooding of arbitrary topography. Journal of hydraulic engineering, 128(3), 289-298.](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%282002%29128%3A3%28289%29)

* [Kim, J., Warnock, A., Ivanov, V. Y., & Katopodes, N. D. (2012).
Coupled modeling of hydrologic and hydrodynamic processes including
overland and channel flow. Advances in water resources, 37, 104-126.](https://www.sciencedirect.com/science/article/pii/S0309170811002211?via%3Dihub)


