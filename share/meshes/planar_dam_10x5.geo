Nx   = 10; // number of cells in X
Ny   = 5;  // number of cells in Y
Nxw  = 2;  // horizontal extent of dam wall in cells (must be even)
Nyw1 = 1;  // vertical extent of upper dam wall in cells
Nyw2 = 2;  // vertical extent of lower dam wall in cells

Lx = 10; // x extent of domain
Ly = 5;  // y extent of domain

dx = Lx/Nx; // quad cell x extent
dy = Ly/Ny; // quad cell y extent

Wx   = Nxw  * dx; // wall thickness (x)
Ltop = Nyw1 * dy; // top wall length (y)
Lbot = Nyw2 * dy; // bottom wall length (y)

// Create the domain using points {pi} connected by line segments {lj}:
//
// p1----l1-----p2    p3-----l5-----p4
// |            |     |             |
// l16          l2    l4            l6
// |            |     |             |
// p5           p6----p7            p8
// |              l3                |
// l15                              l7
// |              l11               |
// p9           p10---p11           p12
// |            |     |             |
// |            |     |             |
// l14          l12   l10           l8
// |            |     |             |
// |            |     |             |
// p13----l13---p14   p15-----l9----p16

// Note that we have to define enough points to break this domain up into
// 7 distinct uniform grids.

Point(1)  = {0, Ly, 0};
Point(2)  = {(Lx-Wx)/2, Ly, 0};
Point(3)  = {(Lx+Wx)/2, Ly, 0};
Point(4)  = {Lx, Ly, 0};
Point(5)  = {0, Ly-Ltop, 0};
Point(6)  = {(Lx-Wx)/2, Ly-Ltop, 0};
Point(7)  = {(Lx+Wx)/2, Ly-Ltop, 0};
Point(8)  = {Lx, Ly-Ltop, 0};
Point(9)  = {0, Lbot, 0};
Point(10) = {(Lx-Wx)/2, Lbot, 0};
Point(11) = {(Lx+Wx)/2, Lbot, 0};
Point(12) = {Lx, Lbot, 0};
Point(13) = {0, 0, 0};
Point(14) = {(Lx-Wx)/2, 0, 0};
Point(15) = {(Lx+Wx)/2, 0, 0};
Point(16) = {Lx, 0, 0};

// Connect the points above with segments and set uniform grid spacing.
// Note that the number of grid points must include endpoints.

Line(1) =  {1, 2};   Transfinite Curve {1}  = (Nx-Nxw)/2+1;
Line(2) =  {2, 6};   Transfinite Curve {2}  = Nyw1+1;
Line(3) =  {6, 7};   Transfinite Curve {3}  = Nxw+1;
Line(4) =  {7, 3};   Transfinite Curve {4}  = Nyw1+1;
Line(5) =  {3, 4};   Transfinite Curve {5}  = (Nx-Nxw)/2+1;
Line(6) =  {4, 8};   Transfinite Curve {6}  = Nyw1+1;
Line(7) =  {8, 12};  Transfinite Curve {7}  = Ny-Nyw1-Nyw2+1;
Line(8) =  {12, 16}; Transfinite Curve {8}  = Nyw2+1;
Line(9) =  {16, 15}; Transfinite Curve {9}  = (Nx-Nxw)/2+1;
Line(10) = {15, 11}; Transfinite Curve {10} = Nyw2+1;
Line(11) = {11, 10}; Transfinite Curve {11} = Nxw+1;
Line(12) = {10, 14}; Transfinite Curve {12} = Nyw2+1;
Line(13) = {14, 13}; Transfinite Curve {13} = (Nx-Nxw)/2+1;
Line(14) = {13, 9};  Transfinite Curve {14} = Nyw2+1;
Line(15) = {9, 5};   Transfinite Curve {15} = Ny-Nyw1-Nyw2+1;
Line(16) = {5, 1};   Transfinite Curve {16} = Nyw1+1;

// In order to define uniform grids, we need four additional horizontal Lines
// and two additional vertical lines that aren't part of the boundary.
Line(17) = {5, 6};   Transfinite Curve {17} = (Nx-Nxw)/2+1;
Line(18) = {7, 8};   Transfinite Curve {18} = (Nx-Nxw)/2+1;
Line(19) = {9, 10};  Transfinite Curve {19} = (Nx-Nxw)/2+1;
Line(20) = {11, 12}; Transfinite Curve {20} = (Nx-Nxw)/2+1;

Line(21) = {6, 10};  Transfinite Curve {21} = Ny-Nyw1-Nyw2+1;
Line(22) = {7, 11};  Transfinite Curve {22} = Ny-Nyw1-Nyw2+1;

// Define the 7 uniform grids by specifying Curves (consisting of line segments)
// that bound Surfaces. A negative line index indicates that the orientation of
// a line is reversed within the curve.

// Upper left grid with corners {p1, p2, p6, p5}
Curve Loop(1) = {1, 2, -17, 16};
Plane Surface(1) = {1};

// Upper right grid with corners {p3, p4, p8, p7}
Curve Loop(2) = {5, 6, -18, 4};
Plane Surface(2) = {2};

// Left grid with corners {p5, p6, p10, p9}
Curve Loop(3) = {17, 21, -19, 15};
Plane Surface(3) = {3};

// Center grid with corners {p6, p7, p11, p10}
Curve Loop(4) = {3, 22, 11, -21};
Plane Surface(4) = {4};

// Right grid with corners {p7, p8, p12, p11}
Curve Loop(5) = {18, 7, -20, -22};
Plane Surface(5) = {5};

// Lower left grid with corners {p9, p10, p14, p13}
Curve Loop(6) = {19, 12, 13, 14};
Plane Surface(6) = {6};

// Lower right grid with corners {p11, p12, p16, p15}
Curve Loop(7) = {20, 8, 9, 10};
Plane Surface(7) = {7};

// Mesh the 7 grids
Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};

// Recombine the grids to produce quads instead of triangles
Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Recombine Surface {6};
Recombine Surface {7};

// Define physical regions in terms of Gmsh Surfaces.

// The "upstream" region consists of grids 1, 3, and 6.
Physical Surface("upstream") = {1, 3, 6};
// The "downstream" region contains all other grids.
Physical Surface("downstream") = {2, 4, 5, 7};

// Define physical surfaces for the domain boundary and dam walls in terms of
// Gmsh Curves.
Physical Curve("boundary") = {1, 5, 6, 7, 8, 9, 13, 14, 15, 16};
Physical Curve("top_wall") = {2, 3, 4};
Physical Curve("bottom_wall") = {2, 3, 4};
