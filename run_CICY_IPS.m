(* Import CICY IPS sampler *)
Get["/home/habjan.e/CY_metric/ips_sampling/PointGeneratorMathematicaCICYIPS.m"]

(* Configure your CICY *)

(* Qunitic Inputs *)
(* dimPs = {4};
coefficients = {{1.0, 1.0, 1.0, 1.0, 1.0, -5.0}};
exponents = {{{5,0,0,0,0}, {0,5,0,0,0}, {0,0,5,0,0}, 
              {0,0,0,5,0}, {0,0,0,0,5}, {1,1,1,1,1}}}; *)

(* Weierstrss Cubic Inputs *)
dimPs = {2};
coefficients = {{1.0, -4.0, 189.07272}};
exponents = {{{1,0,2}, {0,3,0}, {2,1,0}}};

(* Generate points *)
{points, weights, omegas, kappas, {dimCY}} = GeneratePointsMCICYIPS[
    100,      (* total points *)
    5,         (* regions *)
    dimPs,
    coefficients,
    exponents,
    20,        (* precision *)
    1,         (* verbose *)
    True       (* frontEnd *)
];

(* Export results *)
Export["/home/habjan.e/CY_metric/data/ips_mathematica_output/weierstrass_output.mx", {points, weights, omegas, kappas, dimCY}]