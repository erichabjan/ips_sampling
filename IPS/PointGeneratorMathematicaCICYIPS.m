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


(* Compute weights for a point on the CICY *)
getWeightOmegas[varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, g_, eqns_, beqns_, pt_, bpt_, patchLocal_, \[Kappa]_: 1] := Module[
    {patchGlobal, \[Omega], \[Omega]PB, dw, dbw, derivs, bderivs, localCoords, localbCoords, 
    maxPos, maxPosGlobal, goodCoordsIndexSet, \[Omega]Top, OmegaOmegaBar, 
    substCoords, substbCoords, jacRows, i, j, a, b, numEqns, dimCY},
    (

    numEqns = Length[eqns];
    dimCY = Length[varsFlat] - numEqns;
    
    patchGlobal = patchGlobalsFromLocal[patchLocal, dimPs];

    (* Remove patch coordinates *)
    localCoords  = Delete[varsFlat,  List /@ patchGlobal];
    localbCoords = Delete[bvarsFlat, List /@ patchGlobal];

    (* Substitution rules *)
    substCoords  = substRulesBlockwise[varsUnflat, pt, dimPs, patchLocal];
    substbCoords = substRulesBlockwiseBar[bvarsUnflat, bpt, dimPs, patchLocal];
    
    (* Evaluate metric at point *)
    \[Omega] = g /. substCoords /. substbCoords;
    
    (* Compute derivatives of equations *)
    jacRows = {};
    For[i = 1, i <= numEqns, i++,
        derivs = Table[D[eqns[[i]], localCoords[[j]]], {j, Length[localCoords]}] /. substCoords;
        AppendTo[jacRows, derivs];
    ];

    bjacRows = {};
    For[i = 1, i <= numEqns, i++,
        bderivs = Table[
            D[beqns[[i]], localbCoords[[j]]],
            {j, Length[localbCoords]}
        ] /. substbCoords;
        AppendTo[bjacRows, bderivs];
    ];

    (*jacRows  = N[jacRows,  prec];*)
    (*bjacRows = N[bjacRows, prec];*)

    detTol  = 10^-6;       (* numerical threshold for smallest singular value *)
    rankTol = 10^-10;      (* removes any numerical noise before SVD *)

    (* valid if square, full rank, and well-conditioned by svmin *)
    goodBlockQ[mat_] := Module[{sv},
        sv = SingularValueList[Chop[mat, rankTol]];
        If[Length[sv] < Min[Dimensions[mat]], False, Min[sv] > detTol]
    ];

    (* choose k dependent coordinates to find a k x k Jacobian submatrix *)
    maxPoss = {};  (* Local indices of dependent coordinates *)
    maxPossGlobal = {};  (* Global indices of dependent coordinates *)

    If[numEqns <= 3 && Length[localCoords] <= 20,
        (* Exhaustive search: maximize the smallest singular value of J-submatrix *)
        Module[{bestCoords = {}, bestScore = -Infinity, coordCombinations, Jtest, score},
            coordCombinations = Subsets[Range[Length[localCoords]], {numEqns}];
            Do[
                Jtest = Table[jacRows[[alpha, coords[[beta]]]], {alpha, numEqns}, {beta, numEqns}];
                score = If[goodBlockQ[Jtest], Min[SingularValueList[Chop[Jtest, rankTol]]], -Infinity];
                If[score > bestScore, bestScore = score; bestCoords = coords;]
                , {coords, coordCombinations}];
                If[bestScore <= detTol, Return[{Indeterminate, Indeterminate, {}}]];
                maxPoss = bestCoords;
            ],
        (* Greedy with rank test + backtracking lite *)
        Module[{available = Range[Length[localCoords]], chosen = {}, added, sorted, testCoords, Jtest},
            Do[
                added = False;
                (* Heuristic: try columns with larger |∂p_i/∂z_j| first for this equation i *)
                sorted = Sort[available, Abs[jacRows[[i, #1]]] > Abs[jacRows[[i, #2]]] &];
            Do[
                testCoords = Append[chosen, c];
                Jtest = Table[jacRows[[α, testCoords[[β]]]], {α, Length[testCoords]}, {β, Length[testCoords]}];
                If[goodBlockQ[Jtest],
                    AppendTo[chosen, c];
                    available = DeleteCases[available, c];
                    added = True; Break[];
                ];
            , {c, sorted}];
            If[!added, Return[{Indeterminate, Indeterminate, {}}]];
        , {i, 1, numEqns}];
        maxPoss = chosen;
        ]
    ];

    (* Map local indices to global indices in the flattened coord list *)
    maxPossGlobal = Table[Position[varsFlat, localCoords[[maxPoss[[i]]]]][[1, 1]], {i, Length[maxPoss]}];

    (* Build both holomorphic and anti-holomorphic k×k blocks and validate them *)
    jacSubmatrix  = Table[jacRows[[alpha, maxPoss[[beta]]]], {alpha, numEqns}, {beta, numEqns}];
    bjacSubmatrix = Table[bjacRows[[alpha, maxPoss[[beta]]]], {alpha, numEqns}, {beta, numEqns}];

    If[!goodBlockQ[jacSubmatrix] || !goodBlockQ[bjacSubmatrix],
        Return[{Indeterminate, Indeterminate, {}}]
    ];

    jacSubmatrixInv  = Inverse[jacSubmatrix];
    bjacSubmatrixInv = Inverse[bjacSubmatrix];

    (* Build IFT derivative matrix for dependent vs independent chart coords *)
    depLocal = maxPoss; (* indices in localCoords *)
    indLocal = Complement[Range[Length[localCoords]], depLocal];

    Jdep = jacRows[[All, depLocal]];
    Jind = jacRows[[All, indLocal]];
    dDepdIndIFT = -Inverse[Jdep].Jind;  (* dims: k x (n-k) *)

    (* For convenience: map localCoords indices -> global varsFlat indices *)
    localToGlobal = Table[Position[varsFlat, localCoords[[j]]][[1, 1]], {j, Length[localCoords]}];
    depGlobal = localToGlobal[[depLocal]];
    indGlobal = localToGlobal[[indLocal]];

    (* dw[a,i] where i is an intrinsic coordinate index in goodCoordsIndexSet *)
    dw[a_, i_] := Module[{posAdep, posIind, posAind},
    Which[
        MemberQ[patchGlobal, a], 0,

        (* a is one of the independent chart coords *)
        MemberQ[indGlobal, a],
        posAind = First@First@Position[indGlobal, a];
        If[indGlobal[[posAind]] === i, 1, 0],

        (* a is one of the dependent chart coords *)
        MemberQ[depGlobal, a],
        posAdep = First@First@Position[depGlobal, a];
        posIind = Position[indGlobal, i];
        If[posIind === {}, 0, dDepdIndIFT[[posAdep, posIind[[1, 1]]]]],

        True, 0
        ]
        ];
    
    If[!MatrixQ[Jdep, NumericQ] || Det[Jdep] == 0, Return[{Indeterminate, Indeterminate, {}}]];

    (* antiholomorphic IFT derivative matrix *)
    bJdep = bjacRows[[All, depLocal]];
    bJind = bjacRows[[All, indLocal]];
    bdDepdIndIFT = -Inverse[bJdep].bJind;

    If[!MatrixQ[bdDepdIndIFT, NumericQ],
    Return[{Indeterminate, Indeterminate, {}}]
    ];

    If[!MatrixQ[dDepdIndIFT, NumericQ],
    Return[{Indeterminate, Indeterminate, {}}]
    ];

    dbw[a_, i_] := Module[{posAdep, posIind, posAind},
    Which[
        MemberQ[patchGlobal, a], 0,

        MemberQ[indGlobal, a],
        posAind = First@First@Position[indGlobal, a];
        If[indGlobal[[posAind]] === i, 1, 0],

        MemberQ[depGlobal, a],
        posAdep = First@First@Position[depGlobal, a];
        posIind = Position[indGlobal, i];
        If[posIind === {}, 0, bdDepdIndIFT[[posAdep, posIind[[1, 1]]]]],

        True, 0
        ]
        ];

    
    (* Indices of independent coordinates - remove all dependent coords and patch coord *)
    goodCoordsIndexSet = Range[Length[varsFlat]];
    goodCoordsIndexSet = Complement[goodCoordsIndexSet, maxPossGlobal];
    goodCoordsIndexSet = Complement[goodCoordsIndexSet, patchGlobal];

    If[Sort[goodCoordsIndexSet] =!= Sort[indGlobal],
    Return[{Indeterminate, Indeterminate, {}}]
    ];
    
    (* Pull back metric to intrinsic coordinates *)
    \[Omega]PB = Table[
        Sum[dw[a, i] \[Omega][[a, b]] dbw[b, j], 
            {a, Length[varsFlat]}, {b, Length[varsFlat]}],
        {i, goodCoordsIndexSet}, {j, goodCoordsIndexSet}
    ];
    
    (* Top form *)
    \[Omega]Top = Factorial[Length[\[Omega]PB]] Det[\[Omega]PB];
    
    (* Omega wedge Omega-bar factor - determinant of Jacobian submatrices *)
    OmegaOmegaBar = 1/(Det[jacSubmatrix] * Det[bjacSubmatrix]);
    
    (* Weight *)
    w = OmegaOmegaBar / \[Omega]Top;
    
    (* Return[Abs[Chop[{\[Kappa] w, OmegaOmegaBar}]]]; *)
    Return[{Abs[Chop[\[Kappa] w]], Abs[Chop[OmegaOmegaBar]], maxPossGlobal}];
)];

(* Find new Lambda matrices based on min/max weight points *)
GetNewLambdas[varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, eqns_, beqns_, 
    pts_, Ls_, \[Kappa]s_, dimCY_, kahlerModuli_: Automatic] := Module[
    {gFS, allWeights, minData, maxData, minPoint, maxPoint, wMin, wMax, 
    \[Epsilon]Min, \[Epsilon]Max, Px, \[Lambda]Min, \[Lambda]Max, \[Lambda]MinInv, \[Lambda]MaxInv, i},
    (

    Print["GetNewLambdas: Starting with ", Length[pts], " points"];
    Print["GetNewLambdas: Current number of metrics: ", Length[Ls]];
    Print["GetNewLambdas: dimPs = ", dimPs];
    Print["GetNewLambdas: dimCY = ", dimCY];

    (* Precompute standard FS metric once *)
    gFS = getFS[varsUnflat, bvarsUnflat, {}, kahlerModuli];
    
    (* Compute all weights in parallel *)
    allWeights = ParallelTable[
        Module[{pt, patchIndex, wFS, \[CapitalOmega]FS, allMetricWeights, j, g},
            pt = pts[[i, 1]];
            patchLocal = pts[[i, 4]];
            {wFS, \[CapitalOmega]FS, jElimFS} = getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat, dimPs, 
                gFS, eqns, beqns, pt, Conjugate[pt], patchLocal, \[Kappa]s[[1]]];
            
            (* Calculate weights for all metrics *)
            allMetricWeights = Table[
                g = getFS[varsUnflat, bvarsUnflat,
                Table[ConjugateTranspose[Ls[[j]][[k]]] . Ls[[j]][[k]], {k, Length[Ls[[j]]]}],
                kahlerModuli
                ];
                getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat, dimPs,
                g, eqns, beqns, pt, Conjugate[pt], patchLocal, \[Kappa]s[[1]]
                ][[1]],
                {j, Length[Ls]}
                ];
            
            {Min[allMetricWeights], Max[allMetricWeights], pt, wFS^(1/dimCY) - 1, i}
        ],
        {i, Length[pts]},
        DistributedContexts -> Automatic
    ];

    Print["GetNewLambdas: Computed weights for all points"];
    Print["GetNewLambdas: Number of valid weights: ", Count[allWeights, {_?NumericQ, _?NumericQ, __}]];
    Print["GetNewLambdas: Sample of weights (first 3): ", Take[allWeights, Min[3, Length[allWeights]]]];
    
    (* Find global minimum and maximum *)
    minData = SortBy[allWeights, #[[1]] &][[1]];
    maxData = SortBy[allWeights, -#[[2]] &][[1]];
    
    (* Extract data *)
    wMin = minData[[1]];
    minPoint = minData[[3]];
    \[Epsilon]Min = minData[[4]];
    wMax = maxData[[2]];
    maxPoint = maxData[[3]];
    \[Epsilon]Max = maxData[[4]];

    Print["GetNewLambdas: wMin = ", wMin, ", wMax = ", wMax];
    Print["GetNewLambdas: εMin = ", εMin, ", εMax = ", εMax];
    Print["GetNewLambdas: minPoint = ", minPoint];
    Print["GetNewLambdas: maxPoint = ", maxPoint];
    
    (* Construct Lambda matrices for each projective space *)
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

    Print["GetNewLambdas: λMin matrices dimensions: ", Dimensions /@ λMin];
    Print["GetNewLambdas: λMax matrices dimensions: ", Dimensions /@ λMax];
    Print["GetNewLambdas: λMin[[1]] = ", λMin[[1]]];
    Print["GetNewLambdas: λMax[[1]] = ", λMax[[1]]];
    
    (* Hermitianize the inverse matrices *)
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

    Print["GetNewLambdas: λMinInv dimensions: ", Dimensions /@ λMinInv];
    Print["GetNewLambdas: λMaxInv dimensions: ", Dimensions /@ λMaxInv];
    Print["GetNewLambdas: Checking if Cholesky will work..."];
    Print["GetNewLambdas: λMinInv[[1]] numeric? ", And @@ (NumericQ /@ Flatten[λMinInv[[1]]])];
    Print["GetNewLambdas: λMaxInv[[1]] numeric? ", And @@ (NumericQ /@ Flatten[λMaxInv[[1]]])];
    
    Return[{
        Table[CholeskyDecomposition[\[Lambda]MinInv[[i]]], {i, Length[\[Lambda]MinInv]}],
        Table[CholeskyDecomposition[\[Lambda]MaxInv[[i]]], {i, Length[\[Lambda]MaxInv]}]
    }];
)];

(* Patch Normalization Function *)
patchNormalizeFlatPoint[pt_, dimPs_, denomTol_: 10^-30] := Module[
  {blocks, absBlocks, localMaxPos, denoms, normBlocks},
  blocks = splitByBlocks[pt, dimPs];
  absBlocks = Abs /@ blocks;
  localMaxPos = Ordering[#, -1][[1]] & /@ absBlocks;
  denoms = MapThread[#1[[#2]] &, {blocks, localMaxPos}];
  If[Min[Abs[denoms]] < denomTol, Return[$Failed]];
  normBlocks = MapThread[#1/#2 &, {blocks, denoms}];
  Chop @ Flatten[normBlocks]
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
            (* Apply L^{-T} to each sphere point *)
            Table[
                (* Normalize after transformation *)
                Module[{transformed},
                    transformed = Table[
                        Sum[LInvTrans[[a, c]] pointsOnSphere[[j, b, c]], 
                            {c, 1, Length[pointsOnSphere[[j, 1]]]}],
                        {a, 1, Length[pointsOnSphere[[j, 1]]]}
                    ];
                    (* Renormalize to unit sphere *)
                    transformed / Sqrt[Conjugate[transformed] . transformed]
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
SamplePointsIPS[varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, 
    coefficients_, exponents_, L_, numPts_, dimCY_, kahlerModuli_: Automatic, \[Kappa]In_: 1, precision_: 20] := Module[
    {eqns, beqns, numParamsInPn, params, pointsOnSphere, pts, i, j, g, pt, patchIndex, 
    w, \[CapitalOmega], res, allPts, \[Kappa], conf, start, col, totalDeg, numPoints},
    (
    (* Reconstruct equations *)
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

    (* Tolerance *)
    eqnTol = 10^(-Floor[precision/2]);

    (*eqnResidual[pt_List] := Module[{vals},
    vals = (eqns /. Thread[varsFlat -> SetPrecision[pt, precision]]);
    Max[Abs[N[vals, precision]]]
    ];*)

    eqnResidual[pt_List] /; VectorQ[pt, NumericQ] && Length[pt] == Length[varsFlat] := Module[{vals},
    vals = eqns /. Thread[varsFlat -> SetPrecision[pt, precision]];
    Max[Abs[N[vals, precision]]]
    ];

    eqnResidual[_] := Infinity;
    
    (* Determine parameter distribution *)
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
    
    (* Generate points *)

    Clear[t];

    params = Table[Join[{1}, Array[Unique["t"] &, numParamsInPn[[j]]]], {j, Length[numParamsInPn]}];

    ptsGood = {};
    tries = 0;
    maxTries = 200;

    numPoints = Ceiling[numPts / 5];

    While[Length[ptsGood] < numPts && tries < maxTries,
        tries++;
    
        pointsOnSphere = ParallelTable[
            SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
            {i, Length[dimPs]},
            DistributedContexts -> Automatic
        ];

        (* Sample points with metric transformation *)
        ptsBatch = ParallelTable[
            getPointsOnCYIPS[varsUnflat, numParamsInPn, dimPs, params,
                Table[pointsOnSphere[[i, p + (b - 1) numPoints]], 
                    {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
                eqns, L, precision],
            {p, numPoints},
            DistributedContexts -> Automatic
        ];
    
        ptsBatch = Flatten[ptsBatch, 1];
        ptsBatch = Select[ptsBatch, MatchQ[#, {_?NumericQ ..}] && Length[#] == Length[varsFlat] &];
        Print["SamplePointsIPS: Batch ", tries, " produced ", Length[ptsBatch], " raw points."];

        (* Residuals + filtering *)
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
    
    (* Compute metric *)
    g = getFS[varsUnflat, bvarsUnflat, Table[ConjugateTranspose[L[[k]]] . L[[k]], {k, Length[L]}], kahlerModuli];
    
    (* Compute weights *)
    allPts = ParallelTable[
        Module[{point, patchLocal},
        point = pts[[i]];
        patchLocal = patchIndicesByBlock[point, dimPs];
        {w, \[CapitalOmega], jElimGlobal} = getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat, dimPs,
        g, eqns, beqns, point, Conjugate[point], patchLocal, \[Kappa]In];
        Chop[{point, w, \[CapitalOmega], patchLocal, jElimGlobal}]
        ],
        {i, Length[pts]},
        DistributedContexts -> Automatic
    ];
    Print["SamplePointsIPS: Computed weights for ", Length[allPts], " points"];
    Print["SamplePointsIPS: Number with valid weights: ", Count[allPts[[;;, 2]], _?NumericQ]];
    
    (* Estimate kappa if not provided *)
    \[Kappa] = \[Kappa]In;
    If[\[Kappa] == 1,
        \[Kappa] = 1/Mean[allPts[[;;, 2]]];
        allPts[[;;, 2]] = allPts[[;;, 2]] * \[Kappa];
    ];
    
    Return[{allPts, \[Kappa]}];
)];

(* Main function for IPS on CICYs *)
GeneratePointsMCICYIPS[TotalNumPts_, NumRegions_, dimPs_, coefficients_, exponents_, 
    kahlerModuli_: Automatic, precision_: 20, verbose_: 0, frontEnd_: False] := Module[
    {NumPts, varsUnflat, bvarsUnflat, varsFlat, bvarsFlat, eqns, beqns, Ls, allPts, 
    \[Kappa], \[Kappa]s, newPts, r, i, dimCY},
    (
    If[!frontEnd,
        ClientLibrary`SetInfoLogLevel[];
    ];
    
    NumPts = Ceiling[TotalNumPts / NumRegions];
    PrintMsg["Generating " <> ToString[NumPts] <> " points in each of the " <> 
        ToString[NumRegions] <> " regions for a total of " <> ToString[TotalNumPts] <> 
        " points.", frontEnd, verbose];
    
    (* Setup coordinates *)
    varsUnflat = Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
    bvarsUnflat = Table[Subscript[bx, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
    varsFlat = Flatten[varsUnflat];
    bvarsFlat = Flatten[bvarsUnflat];
    
    (* Calculate CICY dimension *)
    dimCY = Plus @@ dimPs - Length[coefficients];
    
    (* Reconstruct equations *)
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
    
    (* Initialize with identity metrics *)
    Ls = {Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]};
    
    PrintMsg["Configuration matrix: " <> ToString[Transpose[
        Table[Module[{start, col, totalDeg},
            start = 1;
            col = {};
            For[i = 1, i <= Length[dimPs], i++,
                totalDeg = Plus @@ exponents[[r, 1, start ;; start + dimPs[[i]]]];
                AppendTo[col, totalDeg];
                start += dimPs[[i]] + 1;
            ];
            col
        ], {r, Length[coefficients]}]
    ]], frontEnd, verbose];
    
    (* Generate initial points with standard FS metric *)
    PrintMsg["Processing region 1", frontEnd, verbose];
    {allPts, \[Kappa]} = SamplePointsIPS[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[-1]], NumPts, dimCY, kahlerModuli, 1, precision];
    \[Kappa]s = {\[Kappa]};
    PrintMsg["Calculated \[Kappa]=" <> ToString[\[Kappa]], frontEnd, verbose];
    
    (* Iteratively generate new regions *)
    For[r = 1, r <= Floor[NumRegions/2], r++,
        Ls = Join[Ls, GetNewLambdas[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, eqns, beqns, allPts, Ls, \[Kappa]s, dimCY, kahlerModuli]];
        
        PrintMsg["Processing region " <> ToString[2*r], frontEnd, verbose];
        {newPts, \[Kappa]} = SamplePointsIPS[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, coefficients, exponents, Ls[[-2]], NumPts, dimCY, kahlerModuli, \[Kappa], precision];
        allPts = Join[allPts, newPts];
        AppendTo[\[Kappa]s, \[Kappa]];
        
        PrintMsg["Processing region " <> ToString[2*r + 1], frontEnd, verbose];
        {newPts, \[Kappa]} = SamplePointsIPS[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, coefficients, exponents, Ls[[-1]], NumPts, dimCY, kahlerModuli, \[Kappa], precision];
        allPts = Join[allPts, newPts];
        AppendTo[\[Kappa]s, \[Kappa]];
    ];
    
    PrintMsg["done.", frontEnd, verbose];
    
    Return[{allPts[[;;, 1]], allPts[[;;, 2]], allPts[[;;, 3]], allPts[[;;, 4]], allPts[[;;, 5]], \[Kappa]s, {dimCY}}];

)];