lc = 0.01;
lc2 = 0.005;
lc3 = 0.01;
L = 1;


Point(1) = {0, 0.1*L, 0, lc};
Point(2) = {0, 0.5*L, 0, lc};
Point(3) = {L, 0.5*L, 0, lc};
Point(4) = {L, 0, 0, lc3};
Point(5) = {0.5*L, 0, 0, lc2};
Point(6) = {0.5*L, 0.1*L, 0, lc2}; 

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};


Line Loop(7) = {2, 3, 4, 5, 6, 1};
Plane Surface(8) = {7};
Plane Surface(9) = {7};
Plane Surface(10) = {7};
Plane Surface(11) = {7};
Plane Surface(12) = {7};
Coherence;
