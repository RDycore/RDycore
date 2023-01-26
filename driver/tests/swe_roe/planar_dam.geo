// This file can be used to generate a mesh representing a dam with two
// barriers in planar (z = 0) geometry using gmsh (https://gmsh.info):
//
// gmsh planar_dam.geo -2
//

// Domain extents
Lx = 200;
Ly = 200;

// Thickness of walls
dx = 10;

// Vertical lengths of top and bottom walls
Ltop = 30;
Lbot = 95;

// Create the domain using points {pi} connected by line segments {lj}:
//
// p1----l1-----p2  p3-----l5-----p4
// |            |   |             |
// |            l2  l4            |
// |            |   |             |
// |            p5--p6            |
// |                              |
// |                              |
// l12                            l6
// |            p7--p8            |
// |            |   |             |
// |            |   |             |
// |            l10 l8            |
// |            |   |             |
// |            |   |             |
// p9----l11----p10 p11-----l7----p12

Point(1) = {0, Ly, 0};
Point(2) = {(Lx-dx)/2, Ly, 0};
Point(3) = {(Lx+dx)/2, Ly, 0};
Point(4) = {Lx, Ly, 0};
Point(5) = {(Lx-dx)/2, Ly-Ltop, 0};
Point(6) = {(Lx+dx)/2, Ly-Ltop, 0};
Point(7) = {(Lx-dx)/2, Lbot, 0};
Point(8) = {(Lx+dx)/2, Lbot, 0};
Point(9) = {0, 0, 0};
Point(10) = {(Lx-dx)/2, 0, 0};
Point(11) = {(Lx+dx)/2, 0, 0};
Point(12) = {Lx, 0, 0};

Line(1) = {1, 2};
Line(2) = {2, 5};
Line(3) = {5, 6};
Line(4) = {6, 3};
Line(5) = {3, 4};
Line(6) = {4, 12};
Line(7) = {12, 11};
Line(8) = {11, 8};
Line(9) = {8, 7};
Line(10) = {7, 10};
Line(11) = {10, 9};
Line(12) = {9, 1};

// Create the domain boundary and, from it, the domain itself.
Curve Loop(1) = {1:12};
Plane Surface(1) = {1};

// Mark the boundary of the domain with a curve constructed from line segments.
Physical Curve("boundary", 1) = {1, 5, 6, 7, 11, 12};

// Mark the top and bottom walls with similar curves.
Physical Curve("top_wall", 2) = {2:4};
Physical Curve("bottom_wall", 3) = {8:10};

// The z=0 plane itself is a surface.
Physical Surface(10) = {1};

