(* Import CICY IPS sampler *)
Get["/home/habjan.e/CY_metric/ips_sampling/IPS/PointGeneratorMathematicaCICYIPS.m"]


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

precisionVal = 20;
verboseVal = 1;
frontEndVal = True;

(* CLI inputs *)
rawArgs = If[ListQ[$CommandLine], $CommandLine, {}];
scriptPos = FirstPosition[rawArgs, s_String /; StringMatchQ[s, ___ ~~ ".m"], Missing["NotFound"]];
args = If[scriptPos === Missing["NotFound"],
  {},
  Drop[rawArgs, scriptPos[[1]]]
];

getRequiredArg[argList_, key_] := Module[{pos},
  pos = FirstPosition[argList, key, Missing["NotFound"]];
  If[pos === Missing["NotFound"] || pos[[1]] >= Length[argList],
    Print["ERROR: Missing required argument: ", key];
    Print["Usage: WolframKernel -script run_quintic.m --output-dir <PATH> --num-regions <INT> --total-points <INT>"];
    Exit[1];
  ];
  argList[[pos[[1]] + 1]]
];

toPositiveInt[x_, key_] := Module[{v = Quiet@Check[ToExpression[x], $Failed]},
  If[!IntegerQ[v] || v < 1,
    Print["ERROR: ", key, " must be a positive integer. Got: ", x];
    Exit[1];
  ];
  v
];

dir = getRequiredArg[args, "--output-dir"];
numRegions = toPositiveInt[getRequiredArg[args, "--num-regions"], "--num-regions"];
totalNumPts = toPositiveInt[getRequiredArg[args, "--total-points"], "--total-points"];

If[!DirectoryQ[dir],
  CreateDirectory[dir, CreateIntermediateDirectories -> True];
  Print["[run_quintic.m] Created directory: ", dir];
];

{points, weights, omegas, patchesLocal, jElimGlobal, kappas, {dimCY}} =
  GeneratePointsMCICYIPS[
    totalNumPts,
    numRegions,
    dimPs,
    coefficients,
    exponents,
    precisionVal,
    verboseVal,
    frontEndVal
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

(* Check if a point is purely numerical *)
NumericPointQ[pt_] := And @@ (NumericQ /@ Flatten[pt]);

(* Flatten per point if nested *)
FlattenPoint[pt_] := Join @@ pt;

(* Convert coefficients to JSON-safe {re,im} objects *)
ComplexToAssoc[z_] := <|"re" -> N[Re[z], 20], "im" -> N[Im[z], 20]|>;

(* Boolean mask for valid points *)
validMask = NumericPointQ /@ pointCoords;

Print["Valid points: ", Count[validMask, True], " out of ", Length[pointCoords]];


If[Count[validMask, True] == 0,

  (* No good points *)
  Print["ERROR: No valid numeric points found!"];
  Print["First few points: ", Take[pointCoords, Min[3, Length[pointCoords]]]],

  (* Else: keep only valid rows *)
  pointCoordsClean    = Pick[pointCoords,    validMask];
  weightsClean        = Pick[weightsData,    validMask];
  omegasClean         = Pick[omegasData,     validMask];
  patchesLocalClean   = Pick[patchesLocal,   validMask];
  jElimGlobalClean    = Pick[jElimGlobal,    validMask];

  (* Derive patch globals from patch locals *)
  patchesGlobalClean = patchGlobalsFromLocal[#, dimPs] & /@ patchesLocalClean;

  (* Force numerical evaluation *)
  pointCoordsNumeric      = N[pointCoordsClean,      20];
  weightsNumeric          = N[weightsClean,          20];
  omegasNumeric           = N[omegasClean,           20];
  kappasNumeric           = N[kappasData,            20];
  patchesLocalNumeric     = N[patchesLocalClean,     20];
  patchesGlobalNumeric    = N[patchesGlobalClean,    20];
  jElimGlobalNumeric      = N[jElimGlobalClean,      20];

  (* Flatten points if nested *)
  pointCoordsNumericFlat =
    If[ArrayDepth[pointCoordsNumeric[[1]]] >= 2,
      FlattenPoint /@ pointCoordsNumeric,
      pointCoordsNumeric
    ];

  (* Split real / imag *)
  coordsReal = N[Re[pointCoordsNumericFlat], 20];
  coordsImag = N[Im[pointCoordsNumericFlat], 20];

  (* Sanity checks *)
  Print["Point shape (real coords): ", Dimensions[coordsReal]];
  Print["Point shape (imag coords): ", Dimensions[coordsImag]];
  Print["First point (real): ", coordsReal[[1]]];
  Print["First point (imag): ", coordsImag[[1]]];
  Print["First weight: ", weightsNumeric[[1]]];
  Print["First omega (expected |Omega|^2): ", omegasNumeric[[1]]];

  (* Export CSV arrays *)

  ptsRealFile = StringTemplate["points_real_``.csv"][numRegions];
  Export[FileNameJoin[{dir, ptsRealFile}], coordsReal];

  ptsImagFile = StringTemplate["points_imag_``.csv"][numRegions];
  Export[FileNameJoin[{dir, ptsImagFile}], coordsImag];

  weightsFile = StringTemplate["weights_``.csv"][numRegions];
  Export[FileNameJoin[{dir, weightsFile}], weightsNumeric];

  omegasFile = StringTemplate["omegas_``.csv"][numRegions];
  Export[FileNameJoin[{dir, omegasFile}], omegasNumeric];

  kappasFile = StringTemplate["kappas_``.csv"][numRegions];
  Export[FileNameJoin[{dir, kappasFile}], kappasNumeric];

  patchesLocalFile = StringTemplate["patches_local_``.csv"][numRegions];
  Export[FileNameJoin[{dir, patchesLocalFile}], patchesLocalNumeric];

  patchesGlobalFile = StringTemplate["patches_global_``.csv"][numRegions];
  Export[FileNameJoin[{dir, patchesGlobalFile}], patchesGlobalNumeric];

  jElimFile = StringTemplate["j_elim_global_``.csv"][numRegions];
  Export[FileNameJoin[{dir, jElimFile}], jElimGlobalNumeric];

  metadataFile = StringTemplate["metadata_``.json"][numRegions];

  metadataAssoc = <|
    "schema_version" -> 1,
    "generator" -> "GeneratePointsMCICYIPS",

    (* Run settings *)
    "requested_total_points" -> totalNumPts,
    "num_regions_requested" -> numRegions,
    "precision" -> precisionVal,
    "verbose" -> verboseVal,
    "front_end" -> frontEndVal,

    (* Post-filter dataset summary *)
    "num_points_valid" -> Length[pointCoordsNumericFlat],
    "point_dimension" -> Length[pointCoordsNumericFlat[[1]]],
    "dim_cy" -> dimCYData,
    "num_hypersurfaces" -> Length[coefficients],
    "num_ambient_factors" -> Length[dimPs],

    (* Geometry definition (for Python reconstruction of CICYPointGenerator) *)
    "dimPs" -> dimPs,
    "ambient" -> dimPs,
    "exponents" -> exponents,
    "coefficients_realimag" -> Map[
      (ComplexToAssoc /@ #) &,
      coefficients
    ],

    (* Physical / numerical conventions *)
    "omega_quantity" -> "|Omega|^2",
    "omega_description" -> "Mathematica omegas CSV stores abs(Omega wedge Omegabar) = |Omega|^2 (real, nonnegative).",
    "weights_quantity" -> "kappa * (|Omega|^2 / top_form_det) with IPS normalization as returned by SamplePointsIPS",
    "patches_local_convention" -> "1-indexed patch index within each projective block (Mathematica indexing)",
    "patches_global_convention" -> "1-indexed flattened global coordinate indices (Mathematica indexing)",
    "j_elim_global_convention" -> "1-indexed flattened global eliminated coordinate indices (Mathematica indexing)",

    (* File manifest *)
    "files" -> <|
      "points_real_csv" -> ptsRealFile,
      "points_imag_csv" -> ptsImagFile,
      "weights_csv" -> weightsFile,
      "omegas_csv" -> omegasFile,
      "kappas_csv" -> kappasFile,
      "patches_local_csv" -> patchesLocalFile,
      "patches_global_csv" -> patchesGlobalFile,
      "j_elim_global_csv" -> jElimFile
    |>
  |>;

  Export[
    FileNameJoin[{dir, metadataFile}],
    metadataAssoc,
    "JSON"
  ];

  Print["Successfully exported ", Length[pointCoordsNumericFlat], " clean points."];
  Print["Metadata written to: ", FileNameJoin[{dir, metadataFile}]];
];