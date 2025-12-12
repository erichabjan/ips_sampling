(* Import CICY IPS sampler *)
Get["/Users/erich/Downloads/Northeastern/IPS_home/ips_sampling/IPS/PointGeneratorMathematicaCICYIPS.m"]

(* Configure your CICY *)

(* Qunitic Inputs *)
(* dimPs = {4};
coefficients = {{1.0, 1.0, 1.0, 1.0, 1.0, -5.0}};
exponents = {{{5,0,0,0,0}, {0,5,0,0,0}, {0,0,5,0,0}, 
              {0,0,0,5,0}, {0,0,0,0,5}, {1,1,1,1,1}}}; *)

(* Weierstrss Cubic Inputs *)
(* dimPs = {2};
coefficients = {{1.0, -4.0, 189.07272}};
exponents = {{{1,0,2}, {0,3,0}, {2,1,0}}}; *)

(* bicubic *)
(* dimPs = {2, 2};
coefficients = {{
  1, 1, 1, 1, 1, 1
}};
exponents = {{
  {3,0,0, 0,3,0},
  {0,3,0, 3,0,0},
  {2,1,0, 1,2,0},
  {1,2,0, 2,1,0},
  {1,0,2, 0,1,2},
  {0,1,2, 1,0,2}
}}; *)

(* split bicubic *)
(* dimPs = {2, 2, 1};
coefficients = {
  {1, 1, 1, 1},
  {1, 1, 1, 1}
};
exponents = {
  {
    {3,0,0, 0,0,0, 1,0},
    {0,3,0, 0,0,0, 0,1},
    {1,1,1, 0,0,0, 1,0},
    {2,1,0, 0,0,0, 0,1}
  },
  {
    {0,0,0, 3,0,0, 1,0},
    {0,0,0, 0,3,0, 0,1},
    {0,0,0, 1,1,1, 1,0},
    {0,0,0, 2,1,0, 0,1}
  }
}; *)

(* multi-equation CICY *)
dimPs = {3, 3};
coefficients = {
  {1, 1, 1, 1},
  {1, 1, 1, 1},
  {1, 1, 1, 1} 
};

exponents = {
  (* Equation 1: cubic in first P^3 only, degree (3,0) *)
  {
    {3, 0, 0, 0,  0, 0, 0, 0},   (* x0^3 *)
    {2, 1, 0, 0,  0, 0, 0, 0},   (* x0^2 x1 *)
    {0, 1, 1, 1,  0, 0, 0, 0},   (* x1 x2 x3 *)
    {0, 0, 3, 0,  0, 0, 0, 0}    (* x2^3 *)
  },

  (* Equation 2: cubic in second P^3 only, degree (0,3) *)
  {
    {0, 0, 0, 0,  3, 0, 0, 0},   (* y0^3 *)
    {0, 0, 0, 0,  2, 1, 0, 0},   (* y0^2 y1 *)
    {0, 0, 0, 0,  0, 1, 1, 1},   (* y1 y2 y3 *)
    {0, 0, 0, 0,  0, 0, 0, 3}    (* y3^3 *)
  },

  (* Equation 3: bilinear, degree (1,1) *)
  {
    {1, 0, 0, 0,  1, 0, 0, 0},   (* x0 y0 *)
    {0, 1, 0, 0,  0, 1, 0, 0},   (* x1 y1 *)
    {0, 0, 1, 0,  0, 0, 1, 0},   (* x2 y2 *)
    {0, 0, 0, 1,  0, 0, 0, 1}    (* x3 y3 *)
  }
};

(* Number of regions *)
numRegions = 11;

(* Generate points *)
{points, weights, omegas, kappas, {dimCY}} = GeneratePointsMCICYIPS[
    20000,      (* total points *)
    numRegions,         (* regions *)
    dimPs,
    coefficients,
    exponents,
    20,        (* precision *)
    1,         (* verbose *)
    True       (* frontEnd *)
];

Print["Checking data structure..."];
Print["Length of points: ", Length[points]];
Print["Head of first point: ", Head[points[[1]]]];
Print["Dimensions of first point: ", Dimensions[points[[1]]]];

pointCoords = points;
weightsData = weights;
omegasData  = omegas;
kappasData  = kappas;
dimCYData   = dimCY;

(*Now force numerical evaluation and remove any bad points*)
Print["Filtering and evaluating points..."];

(*Function to check if a point is purely numerical*)
NumericPointQ[pt_] := And @@ (NumericQ /@ Flatten[pt]);

(* Boolean mask of which points are good *)
validMask = NumericPointQ /@ pointCoords;

Print["Valid points: ", Count[validMask, True], " out of ", Length[pointCoords]];

If[Count[validMask, True] == 0,
  
  (* No good points: just print some diagnostics *)
  Print["ERROR: No valid numeric points found!"];
  Print["First few points: ",
        Take[pointCoords, Min[3, Length[pointCoords]]]],
  
  (* Else: keep only numeric points and corresponding data *)
  pointCoordsClean = Pick[pointCoords, validMask];
  weightsClean     = Pick[weightsData,  validMask];
  omegasClean      = Pick[omegasData,   validMask];

  (* Force numerical evaluation *)
  pointCoordsNumeric = N[pointCoordsClean, 20];
  weightsNumeric     = N[weightsClean,     20];
  omegasNumeric      = N[omegasClean,      20];
  kappasNumeric      = N[kappasData,      20];

  (* Flatten per point if they come as nested lists *)
  FlattenPoint[pt_] := Join @@ pt;
  pointCoordsNumericFlat =
    If[ArrayDepth[pointCoordsNumeric[[1]]] >= 2,
       FlattenPoint /@ pointCoordsNumeric,
       pointCoordsNumeric
    ];

  (* Split real and imaginary parts *)
  coordsReal = N[Re[pointCoordsNumericFlat], 20];
  coordsImag = N[Im[pointCoordsNumericFlat], 20];

  (* Verify they are all numeric *)
  Print["Point shape (real coords): ", Dimensions[coordsReal]];
  Print["First point (real part): ", coordsReal[[1]]];
  Print["First weight: ", weightsNumeric[[1]]];

  dir = "/Users/erich/Downloads/Northeastern/ips_home/Data/ips_output/multi_eq";

  ptsRealFile = StringTemplate["points_real_``.csv"][numRegions];
  Export[FileNameJoin[{dir, ptsRealFile}], coordsReal];

  ptsImgFile = StringTemplate["points_imag_``.csv"][numRegions];
  Export[FileNameJoin[{dir, ptsImgFile}], coordsImag];

  weightsFile = StringTemplate["weights_``.csv"][numRegions];
  Export[FileNameJoin[{dir, weightsFile}], weightsNumeric];

  omegasFile = StringTemplate["omegas_``.csv"][numRegions];
  Export[FileNameJoin[{dir, omegasFile}], omegasNumeric];

  kappasFile = StringTemplate["kappas_``.csv"][numRegions];
  Export[FileNameJoin[{dir, kappasFile}], kappasNumeric];

  (* Export metadata *)
  metafile = StringTemplate["metadata_``.json"][numRegions];
  Export[
    FileNameJoin[{dir, metafile}],
    {
      "dim_cy"          -> dimCYData,
      "num_points"      -> Length[pointCoordsNumeric],
      "num_regions"     -> Length[kappasNumeric],
      "point_dimension" -> Length[pointCoordsNumeric[[1]]]
    },
    "JSON"
  ];

  Print["Successfully exported ", Length[pointCoordsNumeric], " clean points!"];
];