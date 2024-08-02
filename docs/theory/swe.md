# Shallow Water Equations

The two-dimensional shallow water equations can be written in the conservative
form

$$
\frac{\partial\mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = \mathbf{S}\tag{1}\label{1}
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
  Q_i \\[.5em]
  -gh\frac{\partial z}{\partial x} - C_D u \sqrt{u^2 + v^2}\\[.5em]
  -gh\frac{\partial z}{\partial y} - C_D v \sqrt{u^2 + v^2}
  \end{bmatrix}.
\end{align}

In the above expressions,

* $h$ is the flow depth
* $u$ is the vertically-averaged velocity in x-direction
* $v$ is the vertically-averaged velocity in x-direction
* $z$ is the bed elevation
* $C_D = g n^2/h^\frac{1}{3}$ is the drag coefficient
* $n$ is the Manning's coefficient

### Source term

The source term for the momentum equation is

\begin{eqnarray}
\int_{d\Omega} \mathbf{S}_{u_k} d\Omega &=& \int_{\Omega} \left( -gh\frac{\partial z}{\partial x_k} - C_D u \sqrt{u^2 + v^2} \right) d\Omega \nonumber \\
&=& \int_{d\Omega}  -gh\frac{\partial z}{\partial x_k} d\Omega - \int_{\Omega} C_D u \sqrt{u^2 + v^2}  d\Omega 
\end{eqnarray}

where $x_k$ can be either of the spatial coordinates $x$, $y$.

#### Bed slope elevation term

\begin{equation}
\int_{d\Omega}  -gh\frac{\partial z}{\partial x} d\Omega \approx -gh\overline{\frac{\partial z}{\partial x}} \Omega
\end{equation}

For a triangular grid cell

\begin{equation}
\overline{\frac{\partial z}{\partial x}}  = \frac{(y_2 - y_0)(z_1 - z_0) - (y_1 - y_0)(z_2 - z_0)}{(y_2 - y_0)(x_1 - x_0) - (y_1 - y_0)(x_2 - x_0)}
\end{equation}

#### Roughness term

\begin{eqnarray}
\int_{\Omega} C_D u \sqrt{u^2 + v^2}  d\Omega  &\approx& C_D u \sqrt{u^2 + v^2} \Omega
\end{eqnarray}

## Spatial Discretization

We obtain a discrete _finite volume_ method from the shallow water equations by
defining a computational domain $\Omega$ with a boundary $\partial Omega$,
integrating $\ref{swe}$ over this domain, applying Stoke's theorem:

\begin{eqnarray}
\frac{\partial}{\partial t} \int_\Omega \mathbf{U} d\Omega + 
\int_\Omega \frac{\partial\mathbf{E}}{\partial x}  d\Omega + 
\int_\Omega \frac{\partial\mathbf{G}}{\partial y}  d\Omega +  &=&
\int_\Omega \mathbf{S} d\Omega \nonumber\\
\frac{\partial}{\partial t} \int_\Omega \mathbf{U} d\Omega + 
\oint_{d\Omega} \left( \mathbf{E}dy  - \mathbf{G} dx \right) &=&
\int_\Omega \mathbf{S} d\Omega \nonumber\\
 \frac{\partial}{\partial t} \int_\Omega \mathbf{U} d\Omega + 
\int_{d\Omega} \left( \mathbf{F} \cdot \mathbf{n} \right) ds &=&
\int_\Omega \mathbf{S} d\Omega
\end{eqnarray}

In the last step, we have defined a flux vector $\mathbf{F}$ and a unit normal
vector $\mathbf{n}$ pointing outward along the boundary $\partial\Omega$. The
outward normal flux $\mathbf{F \cdot n}$ is given by

\begin{align}
  \mathbf{F.n}
  =
  \begin{bmatrix}
  hu_\perp  \\[.5em]
  huu_\perp + \frac{1}{2}gh^2 \cos\phi + \frac{1}{24}g\left(\Delta h\right)^2\cos\phi\\[.5em]
  hvu_\perp + \frac{1}{2}gh^2 \sin\phi + \frac{1}{24}g\left(\Delta h\right)^2\sin\phi
  \end{bmatrix}
\end{align}

where $u_\perp = u \cos\phi + v \sin\phi$ is the velocity perpendicular to the
boundary and $\phi$ is the angle between the boundary normal vector and the $x$
axis. The surface integral is written in terms of the non-oriented
one-dimensional "surface differential" $ds$.

So far, we have rewritten the shallow water equations in a suggestive but still
continuous form. We can create a discrete representation by partitioning the
domain $\Omega$ into disjoint cells $\{\Omega_i\}$, with $\Omega_i$
corresponding to cell $i$:

$$
\Omega = \bigcup_i \Omega_i
$$

Similarly, we partition the boundary into disjoint line segments representing
boundary faces $\{\partial\Omega_i\}$:

$$
\partial\Omega = \bigcup_i \partial\Omega_i.
$$

With these partitionings, the discrete shallow water equations are

$$
\frac{\partial}{\partial t} \int_{\Omega_i} \mathbf{U} d\Omega_i + 
\int_{d\Omega_i} \left( \mathbf{F} \cdot \mathbf{n} \right) ds =
\int_{\Omega_i}\mathbf{S} d\Omega_i.
$$

### Interface flux
We evaluate the normal flux $\mathbf{F}\cdot\mathbf{n}$ at the (interior) face
shared by cells $i$ and $j$ by computing the _interface flux_ $\mathbf{F}_\perp$
using Roe's method:

\begin{equation}
\mathbf{F.n} \approx \mathbf{F}_\perp^{i,j} =
\frac{1}{2} \left( \mathbf{F}_\perp^{i} + \mathbf{F}_\perp^{j} - \mathbf{\hat{R}} |\mathbf{\hat{\Lambda}| \mathbf{\Delta}\hat{V}} \right).
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
  0                                     & |\hat{u}_\perp| & 0 \\[.5em]  
  0                                     &                         & | \hat{u}_\perp + \hat{a} |^* 
  \end{bmatrix}
\end{align}

with

\begin{eqnarray}
  \hat{h} & = & \sqrt{h_i h_j} \\
  \hat{u} & = & \frac{ \sqrt{h_i} u_i + \sqrt{h_j} u_j}{ \sqrt{h_i} + \sqrt{h_j}} \\
  \hat{v} & = & \frac{ \sqrt{h_i} v_i + \sqrt{h_j} v_j}{ \sqrt{h_i} + \sqrt{h_j}} \\
  \hat{a} & = & \sqrt{\frac{g}{2} \left( h_i + h_j \right)}.
\end{eqnarray}

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
 
## References

* [Bradford, S. F., & Sanders, B. F. (2002). Finite-volume model for shallow-water flooding of arbitrary topography. Journal of hydraulic engineering, 128(3), 289-298.](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%282002%29128%3A3%28289%29)

* [Kim, J., Warnock, A., Ivanov, V. Y., & Katopodes, N. D. (2012).
Coupled modeling of hydrologic and hydrodynamic processes including
overland and channel flow. Advances in water resources, 37, 104-126.](https://www.sciencedirect.com/science/article/pii/S0309170811002211?via%3Dihub)


