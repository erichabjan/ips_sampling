(* Import CICY IPS sampler *)
Get["/Users/erich/Downloads/Northeastern/IPS_home/ips_sampling/IPS/PointGeneratorMathematicaCICYIPS.m"]

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
Print["Checking data structure..."];
Print["Length of points: ", Length[points]];
Print["First element type: ", Head[points[[1]]]];

(*The points variable seems to be the full output*)

(*Check if points is the raw output \
{pts,weights,omegas,kappas,{dimCY}}*)
If[Length[Dimensions[points]] == 1 && 
   Length[points] == 5,(*points is the full output tuple*)
  Print["Points is the full output tuple"];
  pointCoords = points[[1]];
  weightsData = points[[2]];
  omegasData = points[[3]];
  kappasData = points[[4]];
  dimCYData = 
   points[[5, 1]];,(*points is already just the coordinates*)
  Print["Points is coordinate list"];
  pointCoords = points;
  weightsData = weights;
  omegasData = omegas;
  kappasData = kappas;
  dimCYData = dimCY;];

(*Now force numerical evaluation and remove any bad points*)
Print["Filtering and evaluating points..."];

(*Function to check if a point is purely numerical*)
NumericPointQ[pt_] := And @@ (NumericQ /@ Flatten[pt]);

(*Filter to only keep numeric points*)
validIndices = 
  Position[pointCoords, _?NumericPointQ, {1}] // Flatten;
Print["Valid points: ", Length[validIndices], " out of ", 
  Length[pointCoords]];

If[Length[validIndices] == 0, 
  Print["ERROR: No valid numeric points found!"];
  Print["First few points: ", 
   Take[pointCoords, 
    Min[3, Length[
      pointCoords]]]];,(*Extract valid points and corresponding data*)
  pointCoordsClean = pointCoords[[validIndices]];
  weightsClean = weightsData[[validIndices]];
  omegasClean = omegasData[[validIndices]];
  (*Force numerical evaluation*)
  pointCoordsNumeric = N[pointCoordsClean, 20];
  weightsNumeric = N[weightsClean, 20];
  omegasNumeric = N[omegasClean, 20];
  kappasNumeric = N[kappasData, 20];
  (*Verify they are all numeric*)
  Print["Point shape: ", Dimensions[pointCoordsNumeric]];
  Print["First point: ", pointCoordsNumeric[[1]]];
  Print["First weight: ", weightsNumeric[[1]]];
  (*Export*)
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/points_real.csv", 
   Re[pointCoordsNumeric]];
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/points_imag.csv", 
   Im[pointCoordsNumeric]];
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/weights.csv", 
   weightsNumeric];
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/omegas.csv", 
   omegasNumeric];
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/kappas.csv", 
   kappasNumeric];
  (*Export metadata*)
  Export["/Users/erich/Downloads/Northeastern/IPS_home/Data/ips_output/metadata.json", {"dim_cy" ->
      dimCYData, "num_points" -> Length[pointCoordsNumeric], 
    "num_regions" -> Length[kappasNumeric], 
    "point_dimension" -> Length[pointCoordsNumeric[[1]]]}, "JSON"];
  Print["Successfully exported ", Length[pointCoordsNumeric], 
   " clean points!"];];