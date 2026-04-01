(* Improved Point Sampling for CICYs developed by Eric Habjan *)
(* Based on Keller-Lukic 0907.1387 Section 3.2 *)


(* Sample points on sphere *)
SamplePointsOnSphere[dimP_, numPts_] := Module[{randomPoints}, (

    randomPoints = RandomVariate[NormalDistribution[], {numPts, dimP, 2}];
    randomPoints = randomPoints[[;;, ;;, 1]] + I randomPoints[[;;, ;;, 2]];
    randomPoints = Normalize /@ randomPoints;
    Return[randomPoints];

)];


PrintMsg[msg_, frontEnd_, verbose_] := Module[{}, (
    If[verbose > 0,
        If[frontEnd,
            Print[msg];,
            ClientLibrary`SetInfoLogLevel[];
            ClientLibrary`info[msg];
            ClientLibrary`SetErrorLogLevel[];
        ];
    ];
)];


(* Compute Fubini-Study metric with optional Hermitian matrix h *)
getFS[varsUnflat_, bvarsUnflat_, h_: {}, kval_: Automatic] := Module[
  {hh, kk, dimPs, result, i, j, a, b},
(
    dimPs = (Length /@ varsUnflat) - 1;
    hh = h;

    If[hh === {}, hh = Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]];

    kk = Which[
      kval === Automatic, ConstantArray[1, Length[dimPs]],
      NumberQ[kval], ConstantArray[kval, Length[dimPs]],
      ListQ[kval] && Length[kval] == Length[dimPs] && VectorQ[kval, NumericQ], kval,
      True,
        Print["ERROR: kval must be a number or a numeric list of length Length[dimPs]."];
        Return[$Failed]
    ];
    
    (* Build block diagonal metric for product of projective spaces *)
    result = Table[0, {Plus @@ (dimPs + 1)}, {Plus @@ (dimPs + 1)}];
    
    For[i = 1, i <= Length[dimPs], i++,
        For[a = 1, a <= dimPs[[i]] + 1, a++,
            For[b = 1, b <= dimPs[[i]] + 1, b++,
                result[[
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + a,
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + b
                ]] = 1/(\[Pi] kk[[i]]) D[D[Log[bvarsUnflat[[i]] . (hh[[i]] . varsUnflat[[i]])],
                    bvarsUnflat[[i, b]]], varsUnflat[[i, a]]];
            ];
        ];
    ];
    
    Return[result];
)];

(* getAbsMaxPos[alist_] := Module[{k, maxPos}, (
    maxPos = 1;
    For[k = 1, k <= Length[alist], k++,
        If[Abs[alist[[k]]] > Abs[alist[[maxPos]]], maxPos = k];
    ];
    Return[maxPos];
)]; *)

splitByBlocks[pt_, dimPs_] := TakeList[pt, dimPs + 1];

patchIndicesByBlock[pt_, dimPs_] := Module[{blocks, absBlocks},
  blocks = splitByBlocks[pt, dimPs];
  absBlocks = Abs /@ blocks;
  (Ordering[#, -1][[1]] & /@ absBlocks)
];

patchGlobalsFromLocal[patchLocal_, dimPs_] := Module[{offsets},
  offsets = Most @ Accumulate @ Prepend[dimPs + 1, 0];
  offsets + patchLocal
];

substRulesBlockwise[varsUnflat_, pt_, dimPs_, patchLocal_] := Module[
  {blocks, denoms},
  blocks = splitByBlocks[pt, dimPs];
  denoms = MapThread[#1[[#2]] &, {blocks, patchLocal}];
  Flatten@Table[
    Thread[varsUnflat[[i]] -> (blocks[[i]]/denoms[[i]])],
    {i, Length[dimPs]}
  ]
];

substRulesBlockwiseBar[bvarsUnflat_, bpt_, dimPs_, patchLocal_] := Module[
  {blocks, denoms},
  blocks = splitByBlocks[bpt, dimPs];
  denoms = MapThread[#1[[#2]] &, {blocks, patchLocal}];
  Flatten@Table[
    Thread[bvarsUnflat[[i]] -> (blocks[[i]]/denoms[[i]])],
    {i, Length[dimPs]}
  ]
];

(* Choose eliminated coordinates j_elim *)
chooseElimByMaxDQ[eqns_, varsFlat_, pt_, patchGlobal_] := Module[
  {available, dQ, elim = {}, i, j},
  available = ConstantArray[True, Length[varsFlat]];

  available[[patchGlobal]] = False;

  available = available && Thread[Chop[pt - 1] =!= 0];
  For[i = 1, i <= Length[eqns], i++,
    dQ = Abs@Table[D[eqns[[i]], varsFlat[[j]]], {j, Length[varsFlat]}] /. Thread[varsFlat -> pt];

    j = First@Ordering[dQ*Boole[available], -1];
    AppendTo[elim, j];
    available[[j]] = False;
  ];
  elim
];

FSBlockMetric[block_List, volj_?NumericQ, hBlock_: Automatic] := Module[
  {z = N@block, H, hz, s, n},
  n = Length[z];

  H = Which[
    hBlock === Automatic || hBlock === None || hBlock === {}, IdentityMatrix[n],
    MatrixQ[hBlock, NumericQ] && Dimensions[hBlock] === {n, n}, N@hBlock,
    True, Return[ConstantArray[Indeterminate, {n, n}]]
  ];

  hz = H . z;
  s  = Conjugate[z] . hz;

  If[!NumericQ[s] || Abs[s] < 10^-30,
    Return[ConstantArray[Indeterminate, {n, n}]]
  ];

  ((s Transpose[H] - Outer[Conjugate[#1] #2 &, hz, hz])/s^2) * (volj/Pi)
];

FSAmbientMetric[ptFlat_List, dimPs_List, voljs_List, hBlocks_: Automatic] := Module[
  {blocks, mats, hh},
  blocks = splitByBlocks[ptFlat, dimPs];

  hh = Which[
    hBlocks === Automatic || hBlocks === None || hBlocks === {},
      Table[Automatic, {Length[blocks]}],
    ListQ[hBlocks] && Length[hBlocks] == Length[blocks],
      hBlocks,
    True,
      Return[$Failed]
  ];

  mats = MapThread[FSBlockMetric, {blocks, voljs, hh}];
  BlockDiagonalMatrix[mats]
];

metricBlocksFromL[L_List] := Table[
  Chop[ConjugateTranspose[L[[k]]] . L[[k]]],
  {k, Length[L]}
];

prepareWeightEvaluatorCICY[
  varsFlat_, bvarsFlat_, dimPs_, eqns_, numParamsInPn_, hBlocks_: Automatic
] := Module[
  {hh, ts},
  hh = Which[
    hBlocks === Automatic || hBlocks === None || hBlocks === {},
      Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}],
    ListQ[hBlocks] && Length[hBlocks] == Length[dimPs],
      hBlocks,
    True,
      Return[$Failed]
  ];

  ts = Flatten[
    Table[
      ConstantArray[UnitVector[Length[dimPs], i], (dimPs - numParamsInPn)[[i]]],
      {i, Length[dimPs]}
    ],
    1
  ];

  <|
    "dimPs" -> dimPs,
    "nCoords" -> Length[varsFlat],
    "varsFlat" -> varsFlat,
    "numEqns" -> Length[eqns],
    "numParamsInPn" -> numParamsInPn,
    "jacEqns" -> D[eqns, {varsFlat}],
    "ts" -> ts,
    "hBlocks" -> hh
  |>
];


preparePointGeometryCICY[pre_, pt_, patchLocal_] := Module[
  {
    nCoords, numEqns, dimPs,
    patchGlobal, ptNorm,
    jacEvals, jElimGlobal, goodMask, goodCoordsIndexSet,
    Bmat, Amat, detB, dZ, Jpb,
    OmegaOmegaBar
  },

  nCoords = pre["nCoords"];
  numEqns = pre["numEqns"];
  dimPs   = pre["dimPs"];

  patchGlobal = patchGlobalsFromLocal[patchLocal, dimPs];

  ptNorm = patchNormalizeFlatPoint[pt, dimPs];
  If[!VectorQ[ptNorm, NumericQ],
    Return[$Failed]
  ];

  jacEvals = pre["jacEqns"] /. Thread[pre["varsFlat"] -> ptNorm];
  If[!MatrixQ[jacEvals, NumericQ],
    Return[$Failed]
  ];

  jElimGlobal = Module[{available, elim, i, scores, j},
    available = ConstantArray[True, nCoords];
    available[[patchGlobal]] = False;
    available = available && Thread[Chop[ptNorm - 1] =!= 0];
    elim = ConstantArray[0, numEqns];

    For[i = 1, i <= numEqns, i++,
      scores = Abs[jacEvals[[i]]] * Boole[available];
      j = First @ Ordering[scores, -1];
      elim[[i]] = j;
      available[[j]] = False;
    ];
    elim
  ];

  goodMask = ConstantArray[True, nCoords];
  goodMask[[Join[patchGlobal, jElimGlobal]]] = False;
  goodCoordsIndexSet = Pick[Range[nCoords], goodMask];

  Bmat = jacEvals[[All, jElimGlobal]];
  Amat = jacEvals[[All, goodCoordsIndexSet]];

  If[!MatrixQ[Bmat, NumericQ],
    Return[$Failed]
  ];

  detB = Det[Bmat];
  If[!NumericQ[detB] || Abs[detB] < 10^-30,
    Return[$Failed]
  ];

  dZ = -LinearSolve[Bmat, Amat];

  Jpb = ConstantArray[0., {nCoords, Length[goodCoordsIndexSet]}];
  Do[
    Jpb[[goodCoordsIndexSet[[μ]], μ]] = 1.,
    {μ, Length[goodCoordsIndexSet]}
  ];
  Do[
    Jpb[[jElimGlobal[[i]], μ]] = dZ[[i, μ]],
    {i, numEqns}, {μ, Length[goodCoordsIndexSet]}
  ];

  OmegaOmegaBar = Abs[1/detB]^2;

  <|
    "ptRaw" -> pt,
    "ptNorm" -> ptNorm,
    "patchLocal" -> patchLocal,
    "patchGlobal" -> patchGlobal,
    "jacEvals" -> jacEvals,
    "jElimGlobal" -> jElimGlobal,
    "goodCoordsIndexSet" -> goodCoordsIndexSet,
    "Bmat" -> Bmat,
    "Amat" -> Amat,
    "detB" -> detB,
    "dZ" -> dZ,
    "Jpb" -> Jpb,
    "OmegaOmegaBar" -> OmegaOmegaBar
  |>
];

scorePointGeometryWithMetric[pre_, geom_] := Module[
  {ptNorm, Jpb, OmegaOmegaBar, fsPbs, detgNorm, w},

  If[geom === $Failed || !AssociationQ[geom],
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  ptNorm = geom["ptNorm"];
  Jpb = geom["Jpb"];
  OmegaOmegaBar = geom["OmegaOmegaBar"];

  fsPbs = Table[
    With[{gAmbT = FSAmbientMetric[ptNorm, pre["dimPs"], pre["ts"][[k]], pre["hBlocks"]]},
      If[!MatrixQ[gAmbT] || Dimensions[gAmbT] =!= {pre["nCoords"], pre["nCoords"]},
        Return[{Indeterminate, Indeterminate, {}}]
      ];
      Quiet[Check[Transpose[Jpb].gAmbT.Conjugate[Jpb], Indeterminate], Dot::rect]
    ],
    {k, Length[pre["ts"]]}
  ];

  If[!VectorQ[fsPbs, MatrixQ],
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  detgNorm = Chop[LeviCivitaWedgeDet[fsPbs], 10^-30];

  If[!NumericQ[detgNorm] || Abs[detgNorm] < 10^-30,
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  w = Re[OmegaOmegaBar/detgNorm];

  {w, OmegaOmegaBar, geom["jElimGlobal"]}
];

getWeightOmegasPrepared[pre_, pt_, bpt_, patchLocal_] := Module[
  {geom},
  geom = preparePointGeometryCICY[pre, pt, patchLocal];
  scorePointGeometryWithMetric[pre, geom]
];

attachMetricScoresToPointsCICY[pres_List, \[Kappa]s_List, pts_List] := Module[
  {scoredPts},

  scoredPts = ParallelTable[
    Module[
      {
        point, weight, omega, patchLocal, jElimGlobal, regionLabel,
        geom, metricScores, bestMetric, minScore
      },

      point = pts[[i, 1]];
      weight = pts[[i, 2]];
      omega = pts[[i, 3]];
      patchLocal = pts[[i, 4]];
      jElimGlobal = pts[[i, 5]];
      regionLabel = If[Length[pts[[i]]] >= 6, pts[[i, 6]], Missing["NotSpecified"]];

      geom = preparePointGeometryCICY[pres[[1]], point, patchLocal];

      metricScores = Table[
        Module[{tmp},
          tmp = scorePointGeometryWithMetric[pres[[j]], geom];
          If[NumericQ[tmp[[1]]],
            Abs[\[Kappa]s[[j]] tmp[[1]] - 1],
            Infinity
          ]
        ],
        {j, Length[pres]}
      ];

      minScore = Min[metricScores];
      bestMetric = First @ Ordering[metricScores, 1];

      {
        point,
        weight,
        omega,
        patchLocal,
        jElimGlobal,
        regionLabel,
        geom,
        metricScores,
        bestMetric,
        minScore
      }
    ],
    {i, Length[pts]},
    DistributedContexts -> Automatic
  ];

  scoredPts
];

dropCachedMetricDataCICY[ptsWithScores_List] := ptsWithScores[[All, 1 ;; 6]];

selectPointsOwnedByMetricCICY[
  ptsWithScores_List, metricIndex_Integer, ownershipTol_: 10^-6
] := Module[
  {allScores, minScores},
  allScores = ptsWithScores[[All, 8]];
  minScores = ptsWithScores[[All, 10]];
  Table[
    allScores[[i, metricIndex]] <= minScores[[i]] + ownershipTol,
    {i, Length[ptsWithScores]}
  ]
];

PullbackMetric[Jpb_, gAmb_] := Transpose[Jpb].gAmb.Conjugate[Jpb];

LeviCivitaWedgeDet[gpbs_List] := Module[
  {n, perms, signs},

  If[!VectorQ[gpbs, MatrixQ], Return[Indeterminate]];
  n = Length[gpbs];
  If[n < 1, Return[Indeterminate]];

  (* Fast path for CY1 / torus *)
  If[n == 1 && Dimensions[gpbs[[1]]] == {1, 1},
    Return @ N[gpbs[[1, 1, 1]]];
  ];

  (* Fast path for CY3 *)
  If[n == 3 && And @@ (Dimensions[#] == {3, 3} & /@ gpbs),
    perms = Permutations[Range[3]];
    signs = Signature /@ perms;
    Return @ N @ Sum[
      signs[[a]] signs[[b]]
      gpbs[[1, perms[[a, 1]], perms[[b, 1]]]]
      gpbs[[2, perms[[a, 2]], perms[[b, 2]]]]
      gpbs[[3, perms[[a, 3]], perms[[b, 3]]]],
      {a, 6}, {b, 6}
    ];
  ];

  (* Fallback: fully general version *)
  Module[{lcL, lcR, tp, pairs, res},
    lcL = Normal @ LeviCivitaTensor[n];
    lcR = Normal @ LeviCivitaTensor[n];

    tp = TensorProduct @@ Join[gpbs, {lcL, lcR}];

    pairs = Join[
      Table[{2 k - 1, 2 n + k}, {k, 1, n}],
      Table[{2 k, 3 n + k}, {k, 1, n}]
    ];

    res = Quiet[
      Check[TensorContract[tp, pairs], Indeterminate],
      TensorContract::contr
    ];

    If[res === Indeterminate, Return[Indeterminate]];
    res = If[Head[res] === SparseArray, Normal[res], res];
    res = If[ArrayQ[res], First@Flatten[res], res];
    N @ res
  ]
];

(* Patch Normalization Function *)
patchNormalizeFlatPoint[pt_, dimPs_, denomTol_: 10^-30] := Module[
  {blocks, absBlocks, localMaxPos, denoms, normBlocks},
  blocks = splitByBlocks[pt, dimPs];
  absBlocks = Abs /@ blocks;
  localMaxPos = Ordering[#, -1][[1]] & /@ absBlocks;
  denoms = MapThread[#1[[#2]] &, {blocks, localMaxPos}];
  If[Min[Abs[denoms]] < denomTol, Return[{Indeterminate, Indeterminate, {}}]];
  normBlocks = MapThread[#1/#2 &, {blocks, denoms}];
  Chop @ Flatten[normBlocks]
];

(* Compute weights for sampled weights on the CICY *)
getWeightOmegas[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  g_, eqns_, beqns_, pt_, bpt_, patchLocal_, numParamsInPn_, \[Kappa]_: 1
] := Module[
  {hBlocks, pre},

  hBlocks = Which[
    ListQ[g] && Length[g] == Length[dimPs] && And @@ (MatrixQ[#, NumericQ] & /@ g),
      g,
    True,
      Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]
  ];

  pre = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn, hBlocks
  ];

  getWeightOmegasPrepared[pre, pt, bpt, patchLocal]
];

(* Find new Lambda matrices based on min/max weight points *)
GetNewLambdas[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, eqns_, beqns_,
  pts_, Ls_, \[Kappa]s_, dimCY_, numParamsInPn_, kahlerModuli_: Automatic
] := Module[
  {
    preFS, preMetrics, allWeights, minData, maxData, minPoint, maxPoint,
    wMin, wMax, \[Epsilon]Min, \[Epsilon]Max,
    \[Lambda]Min, \[Lambda]Max, \[Lambda]MinInv, \[Lambda]MaxInv
  },

  Print["GetNewLambdas: Starting with ", Length[pts], " points"];
  Print["GetNewLambdas: Current number of metrics: ", Length[Ls]];
  Print["GetNewLambdas: dimPs = ", dimPs];
  Print["GetNewLambdas: dimCY = ", dimCY];

  preFS = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
    Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]
  ];

  preMetrics = Table[
    prepareWeightEvaluatorCICY[
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
      metricBlocksFromL[Ls[[j]]]
    ],
    {j, Length[Ls]}
  ];

  allWeights = ParallelTable[
    Module[{pt, patchLocal, allMetricWeights, chosenWeight, epsChosen},
      pt = pts[[i, 1]];
      patchLocal = pts[[i, 4]];

      allMetricWeights = Table[
        Module[{tmp},
          tmp = getWeightOmegasPrepared[preMetrics[[j]], pt, Conjugate[pt], patchLocal];
          If[NumericQ[tmp[[1]]], \[Kappa]s[[j]] tmp[[1]], Infinity]
        ],
        {j, Length[Ls]}
      ];

      chosenWeight = allMetricWeights[[Ordering[Abs[allMetricWeights - 1], 1][[1]]]];
      epsChosen = chosenWeight^(1/dimCY) - 1;

      {chosenWeight, pt, epsChosen, i}
    ],
    {i, Length[pts]},
    DistributedContexts -> Automatic
  ];

  Print["GetNewLambdas: Computed weights for all points"];
  Print["GetNewLambdas: Number of valid weights: ", Count[allWeights, {_?NumericQ, __}]];
  Print["GetNewLambdas: Sample of weights (first 3): ", Take[allWeights, Min[3, Length[allWeights]]]];

  minData = SortBy[allWeights, #[[1]] &][[1]];
  maxData = SortBy[allWeights, -#[[1]] &][[1]];

  wMin = minData[[1]];
  minPoint = minData[[2]];
  \[Epsilon]Min = minData[[3]];

  wMax = maxData[[1]];
  maxPoint = maxData[[2]];
  \[Epsilon]Max = maxData[[3]];

  Print["GetNewLambdas: wMin = ", wMin, ", wMax = ", wMax];
  Print["GetNewLambdas: εMin = ", \[Epsilon]Min, ", εMax = ", \[Epsilon]Max];
  Print["GetNewLambdas: minPoint = ", minPoint];
  Print["GetNewLambdas: maxPoint = ", maxPoint];

  \[Lambda]Min = Table[
    Module[{minPtBlock, PxBlock},
      minPtBlock = minPoint[[Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + 1 ;;
                             Sum[dimPs[[k]] + 1, {k, 1, i}]]];
      PxBlock = KroneckerProduct[minPtBlock, ConjugateTranspose[minPtBlock]] /
                (ConjugateTranspose[minPtBlock] . minPtBlock);
      1/(1 + \[Epsilon]Min) (IdentityMatrix[dimPs[[i]] + 1] + \[Epsilon]Min PxBlock)
    ],
    {i, Length[dimPs]}
  ];

  \[Lambda]Max = Table[
    Module[{maxPtBlock, PxBlock},
      maxPtBlock = maxPoint[[Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + 1 ;;
                             Sum[dimPs[[k]] + 1, {k, 1, i}]]];
      PxBlock = KroneckerProduct[maxPtBlock, ConjugateTranspose[maxPtBlock]] /
                (ConjugateTranspose[maxPtBlock] . maxPtBlock);
      1/(1 + \[Epsilon]Max) (IdentityMatrix[dimPs[[i]] + 1] + \[Epsilon]Max PxBlock)
    ],
    {i, Length[dimPs]}
  ];

  \[Lambda]MinInv = Table[
    Module[{inv},
      inv = Chop[Inverse[Chop[\[Lambda]Min[[i]]]]];
      Chop[1/2 (inv + ConjugateTranspose[inv])]
    ],
    {i, Length[\[Lambda]Min]}
  ];

  \[Lambda]MaxInv = Table[
    Module[{inv},
      inv = Chop[Inverse[Chop[\[Lambda]Max[[i]]]]];
      Chop[1/2 (inv + ConjugateTranspose[inv])]
    ],
    {i, Length[\[Lambda]Max]}
  ];

  Return[{
    Table[CholeskyDecomposition[\[Lambda]MinInv[[i]]], {i, Length[\[Lambda]MinInv]}],
    Table[CholeskyDecomposition[\[Lambda]MaxInv[[i]]], {i, Length[\[Lambda]MaxInv]}]
  }];
];


(* Sample points on CICY with given metric *)
getPointsOnCYIPS[varsUnflat_, numParamsInPn_, dimPs_, params_, pointsOnSphere_, 
    eqns_, L_, precision_: 20] := Module[
    {subst, pts, i, j, a, b, c, res, maxPoss, absPts, transformedParams, 
     LInvTranspose, transformedSphere, eqSystem, paramVars},
    (
    
    (* For each projective space, compute L^{-T} and apply to sphere points *)
    transformedSphere = Table[
      Module[{LInv, LInvTrans},
          LInv = Inverse[L[[j]]];
          LInvTrans = Transpose[LInv];
          Table[
              Module[{transformed},
                  transformed = Table[
                      Sum[LInvTrans[[a, c]] pointsOnSphere[[j, b, c]],
                          {c, 1, Length[pointsOnSphere[[j, 1]]]}],
                      {a, 1, Length[pointsOnSphere[[j, 1]]]}
                  ];
                  transformed
              ],
              {b, 1, Length[pointsOnSphere[[j]]]}
          ]
        ],
        {j, 1, Length[dimPs]}
    ];
    
    (* Build substitution using transformed sphere points *)
    subst = {};
    For[j = 1, j <= Length[dimPs], j++,
        AppendTo[subst, 
            Table[varsUnflat[[j, a]] -> 
                Sum[params[[j, b]] transformedSphere[[j, b, a]], 
                    {b, Length[params[[j]]]}],
                {a, Length[varsUnflat[[j]]]}
            ]
        ];
    ];
    subst = Flatten[subst];
    
    (* Solve for parameters that put us on the CICY *)
    eqSystem = Table[eqns[[i]] == 0, {i, Length[eqns]}] /. subst;
    eqSystem = SetPrecision[eqSystem, precision];

    paramVars = Flatten[Rest /@ params];

    res = FindInstance[eqSystem, paramVars, Complexes, 1, WorkingPrecision -> precision];
    res = res /. HoldPattern[Null * r_Rule] :> r /. HoldPattern[r_Rule * Null] :> r;
    res = Which[
        res === {} || res === Null || res === $Failed, {},
        MatchQ[res, {{(_Rule | _RuleDelayed) ..} ..}], res,
        MatchQ[res, {(_Rule | _RuleDelayed) ..}], {res},
        True, {}
        ];

    If[res === {} || res === Null, Return[{}]];

    pts = Chop @ N[(varsUnflat /. subst) /. res, precision];
    pts = Map[Flatten, pts];
    pts = Select[pts, VectorQ[#, NumericQ] && Length[#] == Length@Flatten[varsUnflat] &];
    If[pts === {}, Return[{}]];
    
    (* Patch Normalization *)
    pts = Select[patchNormalizeFlatPoint[#, dimPs] & /@ pts, # =!= $Failed &];
    If[pts === {}, Return[{}]];
    
    Return[pts];
)];

(* Sample points and compute weights *)
SamplePointsIPS[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  coefficients_, exponents_, L_, numPts_, dimCY_,
  kahlerModuli_: Automatic, \[Kappa]In_: 1, precision_: 20, regionLabel_: Missing["NotSpecified"]
] := Module[
  {
    eqns, beqns, numParamsInPn, params, pointsOnSphere, pts, i, j,
    w, \[CapitalOmega], allPts, \[Kappa], conf, start, col, totalDeg, numPoints,
    eqnTol, ptsGood, tries, maxTries, ptsBatch, residuals, mask, ptsKeep, nKeep,
    hBlocksNative, preNative
  },

  eqns = Table[
    Sum[coefficients[[i, j]] Times @@ (Power[varsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  beqns = Table[
    Sum[Conjugate[coefficients[[i, j]]] Times @@ (Power[bvarsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  eqnTol = 10^(-Floor[precision/2]);

  eqnResidual[pt_List] /; VectorQ[pt, NumericQ] && Length[pt] == Length[varsFlat] := Module[{vals},
    vals = eqns /. Thread[varsFlat -> SetPrecision[pt, precision]];
    Max[Abs[N[vals, precision]]]
  ];
  eqnResidual[_] := Infinity;

  conf = {};
  For[i = 1, i <= Length[coefficients], i++,
    start = 1;
    col = {};
    For[j = 1, j <= Length[dimPs], j++,
      totalDeg = Plus @@ exponents[[i, 1, start ;; start + dimPs[[j]]]];
      AppendTo[col, totalDeg];
      start += dimPs[[j]] + 1;
    ];
    AppendTo[conf, col];
  ];

  numParamsInPn = Table[1, {i, Length[dimPs]}];
  For[i = 1, i <= Length[eqns] - Length[dimPs], i++,
    If[Union[numParamsInPn*conf[[i]]] === {0},
      numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
    ];
  ];

  While[Length[eqns] > Plus @@ numParamsInPn,
    For[i = 1, i <= Length[eqns], i++,
      If[Length[eqns] == Plus @@ numParamsInPn, Break[];];
      numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
    ];
  ];

  While[Length[eqns] < Plus @@ numParamsInPn,
    For[i = 1, i <= Length[numParamsInPn], i++,
      numParamsInPn[[i]]--;
      If[Min[Transpose[conf . numParamsInPn]] == 0,
        numParamsInPn[[i]]++;,
        Break[];
      ];
    ];
  ];

  i = 1;
  While[i <= Length[numParamsInPn],
    If[numParamsInPn[[i]] > dimPs[[i]],
      For[j = 1, j <= Length[numParamsInPn], j++,
        If[numParamsInPn[[j]] >= dimPs[[j]],
          Continue[];,
          numParamsInPn[[j]]++; numParamsInPn[[i]]--; Break[];
        ];
      ];
      Continue[];
    ];
    i++;
  ];

  hBlocksNative = metricBlocksFromL[L];
  preNative = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn, hBlocksNative
  ];

  Clear[t];
  params = Table[Join[{1}, Array[Unique["t"] &, numParamsInPn[[j]]]], {j, Length[numParamsInPn]}];

  ptsGood = {};
  tries = 0;
  maxTries = 200;
  numPoints = Ceiling[numPts/5];

  While[Length[ptsGood] < numPts && tries < maxTries,
    tries++;

    pointsOnSphere = ParallelTable[
      SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
      {i, Length[dimPs]},
      DistributedContexts -> Automatic
    ];

    ptsBatch = ParallelTable[
      getPointsOnCYIPS[
        varsUnflat, numParamsInPn, dimPs, params,
        Table[
          pointsOnSphere[[i, p + (b - 1) numPoints]],
          {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}
        ],
        eqns, L, precision
      ],
      {p, numPoints},
      DistributedContexts -> Automatic
    ];

    ptsBatch = Flatten[ptsBatch, 1];
    ptsBatch = Select[ptsBatch, MatchQ[#, {_?NumericQ ..}] && Length[#] == Length[varsFlat] &];
    Print["SamplePointsIPS: Batch ", tries, " produced ", Length[ptsBatch], " raw points."];

    residuals = eqnResidual /@ ptsBatch;
    mask = Thread[residuals < eqnTol];

    ptsKeep = Pick[ptsBatch, mask, True];
    nKeep = Length[ptsKeep];

    ptsGood = Join[ptsGood, ptsKeep];

    Print["SamplePointsIPS: Batch ", tries, " kept ", nKeep,
      " points (eqnTol=", eqnTol, "). Total kept = ", Length[ptsGood], "."];
  ];

  pts = ptsGood;
  Print["SamplePointsIPS: Final valid points = ", Length[pts], " (requested ", numPts, ")."];

  allPts = ParallelTable[
    Module[{point, patchLocal},
      point = pts[[i]];
      patchLocal = patchIndicesByBlock[point, dimPs];
      {w, \[CapitalOmega], jElimGlobal} =
        getWeightOmegasPrepared[preNative, point, Conjugate[point], patchLocal];
      Chop[{point, w, \[CapitalOmega], patchLocal, jElimGlobal, regionLabel}]
    ],
    {i, Length[pts]},
    DistributedContexts -> Automatic
  ];

  allPts = Select[
    allPts,
    NumericQ[#[[2]]] && NumericQ[#[[3]]] &&
    Abs[#[[2]]] < Infinity && Abs[#[[3]]] < Infinity &
  ];

  If[Length[allPts] == 0,
    Print["ERROR: All weights invalid; returning empty set."];
    Return[{{}, Indeterminate, numParamsInPn}];
  ];

  Print["SamplePointsIPS: Computed weights for ", Length[allPts], " points"];
  Print["SamplePointsIPS: Number with valid weights: ", Count[allPts[[;;, 2]], _?NumericQ]];

  \[Kappa] = \[Kappa]In;
  If[\[Kappa] == 1,
    \[Kappa] = 1/Mean[allPts[[;;, 2]]];
  ];

  Return[{allPts, \[Kappa], numParamsInPn}];
];


(* Rejection Sampling *)
SamplePointsWithRejectionCICY[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  coefficients_, exponents_, Ls_, pres_, LPos_, numPts_, dimCY_, numParamsInPn_,
  \[Kappa]s_, kahlerModuli_: Automatic, precision_: 20,
  startPts_: {}, numSampledInitial_: 0,
  frontEnd_: False, verbose_: 0
] := Module[
  {
    newPts, pts, ptsScored, resampleCounter, numSampled, numAccepted,
    numToSample, tmpPts, tmpKappa, tmpNumParams, keepMask, keptPts,
    ownershipTol, nNeeded, estAcceptance, safetyFactor,
    minBatch, maxBatch, acceptedThisPass
  },

  ownershipTol = 10^-6;
  resampleCounter = 0;
  numSampled = 0;
  numAccepted = 0;
  newPts = {};

  safetyFactor = 1.25;
  minBatch = 250;
  maxBatch = 20000;

  While[Length[newPts] < numPts,

    nNeeded = numPts - Length[newPts];

    If[startPts =!= {} && resampleCounter == 0,
      pts = startPts;
      numSampled = numSampledInitial;,
      
      estAcceptance = If[numSampled > 0 && numAccepted > 0,
        N[numAccepted/numSampled],
        1./Max[1, Length[Ls]]
      ];

      estAcceptance = Max[10^-3, estAcceptance];

      numToSample = Ceiling[safetyFactor * nNeeded / estAcceptance];
      numToSample = Max[minBatch, numToSample];
      numToSample = Min[maxBatch, numToSample];

      {tmpPts, tmpKappa, tmpNumParams} = SamplePointsIPS[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[LPos]],
        numToSample, dimCY, kahlerModuli, \[Kappa]s[[LPos]], precision, LPos
      ];

      pts = tmpPts;
      numSampled += Length[pts];
    ];

    ptsScored = attachMetricScoresToPointsCICY[pres, \[Kappa]s, pts];

    keepMask = selectPointsOwnedByMetricCICY[ptsScored, LPos, ownershipTol];
    keptPts = Pick[ptsScored, keepMask, True];
    acceptedThisPass = Length[keptPts];

    numAccepted += acceptedThisPass;

    newPts = Join[newPts, dropCachedMetricDataCICY[keptPts]];

    PrintMsg[
      "Region " <> ToString[LPos] <> ": kept " <> ToString[Length[newPts]] <>
      "/" <> ToString[numPts] <> " points after rejection pass " <>
      ToString[resampleCounter + 1] <>
      " (accepted this pass = " <> ToString[acceptedThisPass] <>
      ", estimated acceptance ≈ " <> ToString[N[If[numSampled > 0, numAccepted/numSampled, 0], 4]] <> ").",
      frontEnd, verbose
    ];

    resampleCounter++;
  ];

  PrintMsg[
    "Region " <> ToString[LPos] <> ": acceptance = " <>
    ToString[numAccepted] <> "/" <> ToString[numSampled] <> ".",
    frontEnd, verbose
  ];

  Return[{newPts, numAccepted, numSampled}];
];

(* Main function for IPS on CICYs *)
GeneratePointsMCICYIPS[
  TotalNumPts_, NumRegions_, dimPs_, coefficients_, exponents_,
  kahlerModuli_: Automatic, precision_: 20, verbose_: 0, frontEnd_: False
] := Module[
  {
    NumPts, varsUnflat, bvarsUnflat, varsFlat, bvarsFlat, eqns, beqns,
    Ls, allPts, regionPts, newPts, pres,
    \[Kappa], \[Kappa]s, numParamsInPn, dimCY,
    r, i, Acceptance, NumSample, Acceptances, NumSamples,
    regionLabels, newLambdaPair, nextRegionIndex
  },

  If[!frontEnd, ClientLibrary`SetInfoLogLevel[]];

  NumPts = Ceiling[TotalNumPts / NumRegions];
  PrintMsg[
    "Generating " <> ToString[NumPts] <> " points in each of the " <>
    ToString[NumRegions] <> " regions for a total of " <> ToString[TotalNumPts] <> " points.",
    frontEnd, verbose
  ];

  varsUnflat = Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
  bvarsUnflat = Table[Subscript[bx, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
  varsFlat = Flatten[varsUnflat];
  bvarsFlat = Flatten[bvarsUnflat];

  dimCY = Plus @@ dimPs - Length[coefficients];

  eqns = Table[
    Sum[coefficients[[i, j]] Times @@ (Power[varsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  beqns = Table[
    Sum[Conjugate[coefficients[[i, j]]] Times @@ (Power[bvarsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  Ls = {Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]};

  PrintMsg["Configuration matrix: " <> ToString[Transpose[
    Table[
      Module[{start, col, totalDeg},
        start = 1;
        col = {};
        For[i = 1, i <= Length[dimPs], i++,
          totalDeg = Plus @@ exponents[[r, 1, start ;; start + dimPs[[i]]]];
          AppendTo[col, totalDeg];
          start += dimPs[[i]] + 1;
        ];
        col
      ],
      {r, Length[coefficients]}
    ]
  ]], frontEnd, verbose];

  regionPts = {};
  Acceptances = {};
  NumSamples = {};

  PrintMsg["Processing region 1", frontEnd, verbose];
  {newPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
    varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
    dimPs, coefficients, exponents, Ls[[1]],
    NumPts, dimCY, kahlerModuli, 1, precision, 1
  ];
  regionPts = {newPts};
  \[Kappa]s = {\[Kappa]};
  Acceptances = {Length[newPts]};
  NumSamples = {Length[newPts]};
  PrintMsg["Calculated \[Kappa] = " <> ToString[\[Kappa]], frontEnd, verbose];

  nextRegionIndex = 2;

  While[nextRegionIndex <= NumRegions,

    newLambdaPair = GetNewLambdas[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, eqns, beqns, Flatten[regionPts, 1],
      Ls, \[Kappa]s, dimCY, numParamsInPn, kahlerModuli
    ];

    (* First new metric in the pair *)
    Ls = Append[Ls, newLambdaPair[[1]]];

    PrintMsg["Processing region " <> ToString[nextRegionIndex], frontEnd, verbose];
    {newPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, coefficients, exponents, Ls[[-1]],
      NumPts, dimCY, kahlerModuli, 1, precision, nextRegionIndex
    ];
    AppendTo[regionPts, newPts];
    AppendTo[\[Kappa]s, \[Kappa]];
    AppendTo[Acceptances, Length[newPts]];
    AppendTo[NumSamples, Length[newPts]];

    nextRegionIndex++;

    (* Second new metric in the pair, only if needed *)
    If[nextRegionIndex <= NumRegions,
      Ls = Append[Ls, newLambdaPair[[2]]];

      PrintMsg["Processing region " <> ToString[nextRegionIndex], frontEnd, verbose];
      {newPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[-1]],
        NumPts, dimCY, kahlerModuli, 1, precision, nextRegionIndex
      ];
      AppendTo[regionPts, newPts];
      AppendTo[\[Kappa]s, \[Kappa]];
      AppendTo[Acceptances, Length[newPts]];
      AppendTo[NumSamples, Length[newPts]];

      nextRegionIndex++;
    ];
  ];

  pres = Table[
    prepareWeightEvaluatorCICY[
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
      metricBlocksFromL[Ls[[j]]]
    ],
    {j, Length[Ls]}
  ];


  If[Length[Ls] > 1,
    pres = Table[
      prepareWeightEvaluatorCICY[
        varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
        metricBlocksFromL[Ls[[j]]]
      ],
      {j, Length[Ls]}
    ];

    PrintMsg["Now revisiting all regions with full rejection sampling.", frontEnd, verbose];
    For[r = 1, r <= Length[Ls], r++,
      PrintMsg["Reprocessing region " <> ToString[r], frontEnd, verbose];

      {newPts, Acceptance, NumSample} = SamplePointsWithRejectionCICY[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents,
        Ls, pres, r, NumPts, dimCY, numParamsInPn, \[Kappa]s,
        kahlerModuli, precision,
        regionPts[[r]], NumSamples[[r]],
        frontEnd, verbose
      ];

      regionPts[[r]] = newPts;
      Acceptances[[r]] = Acceptance;
      NumSamples[[r]] = NumSample;
    ];
  ];

  regionPts = Table[
    Module[{nHere = Length[regionPts[[i]]], take},
      take = Min[NumPts, nHere];
      If[nHere <= take,
        regionPts[[i]],
        RandomSample[regionPts[[i]], take]
      ]
    ],
    {i, Length[regionPts]}
  ];

  \[Kappa]s = Table[
    If[Length[regionPts[[i]]] > 0,
      1/Mean[regionPts[[i, ;;, 2]]],
      1
    ],
    {i, Length[regionPts]}
  ];

  regionLabels = Flatten@Table[
    ConstantArray[r, Length[regionPts[[r]]]],
    {r, Length[regionPts]}
  ];

  allPts = Flatten[regionPts, 1];

  PrintMsg["done.", frontEnd, verbose];

  Return[{
    allPts[[;;, 1]],
    allPts[[;;, 2]],
    allPts[[;;, 3]],
    allPts[[;;, 4]],
    allPts[[;;, 5]],
    regionLabels,
    \[Kappa]s,
    Acceptances,
    NumSamples,
    {dimCY},
    Ls
  }];
];