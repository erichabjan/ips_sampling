(* ::Package:: *)

(* Improved Point Sampling for CICYs *)
(* Based on Keller-Lukic 0907.1387 Section 3.2 *)

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
getFS[varsUnflat_, bvarsUnflat_, h_: {}, kval_: 1] := Module[{hh, s, bs, kk, dimPs, result, i, j, a, b}, (
    dimPs = (Length /@ varsUnflat) - 1;
    hh = h;
    kk = kval;
    
    If[hh === {}, hh = Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]];
    If[!IntegerQ[kk], kk = 1];
    
    (* Build block diagonal metric for product of projective spaces *)
    result = Table[0, {Plus @@ (dimPs + 1)}, {Plus @@ (dimPs + 1)}];
    
    For[i = 1, i <= Length[dimPs], i++,
        For[a = 1, a <= dimPs[[i]] + 1, a++,
            For[b = 1, b <= dimPs[[i]] + 1, b++,
                result[[
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + a,
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + b
                ]] = 1/(\[Pi] kk) D[D[Log[bvarsUnflat[[i]] . (hh[[i]] . varsUnflat[[i]])], 
                    bvarsUnflat[[i, b]]], varsUnflat[[i, a]]];
            ];
        ];
    ];
    
    Return[result];
)];

getAbsMaxPos[alist_] := Module[{k, maxPos}, (
    maxPos = 1;
    For[k = 1, k <= Length[alist], k++,
        If[Abs[alist[[k]]] > Abs[alist[[maxPos]]], maxPos = k];
    ];
    Return[maxPos];
)];

(* Compute weights for a point on the CICY *)
getWeightOmegas[varsFlat_, bvarsFlat_, varsUnflat_, dimPs_, g_, eqns_, beqns_, pt_, bpt_, patchIndex_, \[Kappa]_: 1] := Module[
    {\[Omega], \[Omega]PB, dw, dbw, derivs, bderivs, localCoords, localbCoords, 
    maxPos, maxPosGlobal, goodCoordsIndexSet, \[Omega]Top, OmegaOmegaBar, 
    substCoords, substbCoords, jacRows, i, j, a, b, numEqns, dimCY},
    (
    numEqns = Length[eqns];
    dimCY = Length[varsFlat] - numEqns;
    
    (* Remove patch coordinate *)
    localCoords = Delete[varsFlat, patchIndex];
    localbCoords = Delete[bvarsFlat, patchIndex];
    
    (* Substitution rules *)
    substCoords = Table[varsFlat[[k]] -> pt[[k]]/pt[[patchIndex]], {k, Length[pt]}];
    substbCoords = Table[bvarsFlat[[k]] -> bpt[[k]]/bpt[[patchIndex]], {k, Length[bpt]}];
    
    (* Evaluate metric at point *)
    \[Omega] = g /. substCoords /. substbCoords;
    
    (* Compute derivatives of equations *)
    jacRows = {};
    For[i = 1, i <= numEqns, i++,
        derivs = Table[D[eqns[[i]], localCoords[[j]]], {j, Length[localCoords]}] /. substCoords;
        AppendTo[jacRows, derivs];
    ];

    (* Find k coordinates to eliminate (one per equation) *)
    (* We need to find k coordinates such that the k×k Jacobian submatrix is non-singular *)
    maxPoss = {};  (* Local indices of dependent coordinates *)
    maxPossGlobal = {};  (* Global indices of dependent coordinates *)
    availableCoords = Range[Length[localCoords]];

    For[i = 1, i <= numEqns, i++,
        (* Find coordinate with largest derivative in equation i among available coords *)
        bestPos = 0;
        bestVal = 0;
        For[j = 1, j <= Length[availableCoords], j++,
            If[Abs[jacRows[[i, availableCoords[[j]]]]] > bestVal,
                bestVal = Abs[jacRows[[i, availableCoords[[j]]]]];
                bestPos = availableCoords[[j]];
            ];
        ];
        AppendTo[maxPoss, bestPos];
        AppendTo[maxPossGlobal, Position[varsFlat, localCoords[[bestPos]]][[1, 1]]];
        availableCoords = DeleteCases[availableCoords, bestPos];
    ];
    
    (* Compute antiholomorphic Jacobian for all equations *)
    bjacRows = {};
    For[i = 1, i <= numEqns, i++,
        bderivs = Table[D[beqns[[i]], localbCoords[[j]]], {j, Length[localbCoords]}] /. substbCoords;
        AppendTo[bjacRows, bderivs];
    ];
    
    (* Build Jacobian submatrix for dependent coordinates *)
    (* dw[a,i] gives derivative of ambient coordinate a with respect to intrinsic coordinate i *)
    jacSubmatrix = Table[jacRows[[alpha, maxPoss[[beta]]]], {alpha, numEqns}, {beta, numEqns}];
    jacSubmatrixInv = Inverse[jacSubmatrix];

    (* DEBUG: Check if Jacobian is singular *)
    If[!And @@ (NumericQ /@ Flatten[jacSubmatrixInv]),
        Return[{Indeterminate, Indeterminate}];
    ];

    dw[a_, i_] := If[a == patchIndex, 
        0,
        If[MemberQ[maxPossGlobal, a],
            (* This is a dependent coordinate: z^a = z^{dep_alpha} *)
            Module[{alphaIndex, localIndexI},
                (* Find which dependent coordinate this is (1 to k) *)
                alphaIndex = Position[maxPossGlobal, a][[1, 1]];
            
                (* Check if i is also a dependent coordinate or independent *)
                If[MemberQ[maxPossGlobal, i],
                    (* i is dependent - need more complex formula *)
                    (* For dependent coords, ∂z^{dep_alpha}/∂z^{dep_beta} involves Jacobian *)
                    Module[{betaIndex},
                        betaIndex = Position[maxPossGlobal, i][[1, 1]];
                        (* This should rarely be called for standard intrinsic coordinates *)
                        (* but we include it for completeness *)
                        If[alphaIndex == betaIndex, 
                            1, 
                            0
                        ]
                    ],
                    (* i is independent - apply IFT: ∂z^{dep_alpha}/∂z^i = -[J^{-1}]_{alpha,gamma} ∂p_gamma/∂z^i *)
                    -Sum[jacSubmatrixInv[[alphaIndex, gamma]] * D[eqns[[gamma]], varsFlat[[i]]], {gamma, numEqns}] /. substCoords
                ]
            ],
            (* This is an independent coordinate *)
            If[a == i, 1, 0]
        ]
    ];
    
    (* Build antiholomorphic Jacobian submatrix *)
    bjacSubmatrix = Table[bjacRows[[alpha, maxPoss[[beta]]]], {alpha, numEqns}, {beta, numEqns}];
    bjacSubmatrixInv = Inverse[bjacSubmatrix];

    (* DEBUG: Check if antiholomorphic Jacobian is singular *)
    If[!And @@ (NumericQ /@ Flatten[bjacSubmatrixInv]),
        Return[{Indeterminate, Indeterminate}];
    ];

    (* dbw[a,i] gives ∂\bar{z}^a/∂\bar{w}^i *)
    dbw[a_, i_] := If[a == patchIndex, 
        0,
        If[MemberQ[maxPossGlobal, a],
            (* This is a dependent coordinate *)
            Module[{alphaIndex},
                alphaIndex = Position[maxPossGlobal, a][[1, 1]];
            
                If[MemberQ[maxPossGlobal, i],
                    (* i is dependent *)
                    Module[{betaIndex},
                        betaIndex = Position[maxPossGlobal, i][[1, 1]];
                        If[alphaIndex == betaIndex, 
                            1, 
                            0
                        ]
                    ],
                    (* i is independent - apply IFT *)
                    -Sum[bjacSubmatrixInv[[alphaIndex, gamma]] * D[beqns[[gamma]], bvarsFlat[[i]]], {gamma, numEqns}] /. substbCoords
                ]
            ],
            (* This is an independent coordinate *)
            If[a == i, 1, 0]
        ]
    ];
    
    (* Indices of independent coordinates - remove all dependent coords and patch coord *)
    goodCoordsIndexSet = Table[i, {i, Length[varsFlat]}];
    For[i = 1, i <= Length[maxPossGlobal], i++,
        goodCoordsIndexSet = DeleteCases[goodCoordsIndexSet, maxPossGlobal[[i]]];
    ];
    goodCoordsIndexSet = DeleteCases[goodCoordsIndexSet, patchIndex];
    
    (* Pull back metric to intrinsic coordinates *)
    \[Omega]PB = Table[
        Sum[dw[a, i] \[Omega][[a, b]] dbw[b, j], 
            {a, Length[varsFlat]}, {b, Length[varsFlat]}],
        {i, goodCoordsIndexSet}, {j, goodCoordsIndexSet}
    ];

    (* DEBUG: Check pullback metric *)
    If[!And @@ (NumericQ /@ Flatten[\[Omega]PB]),
        Return[{Indeterminate, Indeterminate}];
    ];
    
    (* Top form *)
    \[Omega]Top = Factorial[Length[\[Omega]PB]] Det[\[Omega]PB];
    
    (* Omega wedge Omega-bar factor - determinant of Jacobian submatrices *)
    OmegaOmegaBar = 1/(Det[jacSubmatrix] * Det[bjacSubmatrix]);
    
    (* Weight *)
    w = OmegaOmegaBar / \[Omega]Top;
    
    Return[Abs[Chop[{\[Kappa] w, OmegaOmegaBar}]]];
)];

(* Find new Lambda matrices based on min/max weight points *)
GetNewLambdas[varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, eqns_, beqns_, 
    pts_, Ls_, \[Kappa]s_, dimCY_] := Module[
    {gFS, allWeights, minData, maxData, minPoint, maxPoint, wMin, wMax, 
    \[Epsilon]Min, \[Epsilon]Max, Px, \[Lambda]Min, \[Lambda]Max, \[Lambda]MinInv, \[Lambda]MaxInv, i},
    (

    Print["GetNewLambdas: Starting with ", Length[pts], " points"];
    Print["GetNewLambdas: Current number of metrics: ", Length[Ls]];
    Print["GetNewLambdas: dimPs = ", dimPs];
    Print["GetNewLambdas: dimCY = ", dimCY];

    (* Precompute standard FS metric once *)
    gFS = getFS[varsUnflat, bvarsUnflat];
    
    (* Compute all weights in parallel *)
    allWeights = ParallelTable[
        Module[{pt, patchIndex, wFS, \[CapitalOmega]FS, allMetricWeights, j, g},
            pt = pts[[i, 1]];
            patchIndex = pts[[i, 4]];
            {wFS, \[CapitalOmega]FS} = getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, dimPs, 
                gFS, eqns, beqns, pt, Conjugate[pt], patchIndex, \[Kappa]s[[1]]];
            
            (* Calculate weights for all metrics *)
            allMetricWeights = Table[
                g = getFS[varsUnflat, bvarsUnflat, Table[ConjugateTranspose[Ls[[j]][[k]]] . Ls[[j]][[k]], {k, Length[Ls[[j]]]}]];
                First[getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, dimPs, 
                    g, eqns, beqns, pt, Conjugate[pt], patchIndex, \[Kappa]s[[1]]]],
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

(* Sample points on CICY with given metric *)
getPointsOnCYIPS[varsUnflat_, numParamsInPn_, dimPs_, params_, pointsOnSphere_, 
    eqns_, L_, precision_: 20] := Module[
    {subst, pts, i, j, a, b, res, maxPoss, absPts, transformedSphere},
    (
    Print["getPointsOnCYIPS: L structure: ", Dimensions /@ L];
    Print["getPointsOnCYIPS: pointsOnSphere structure: ", Dimensions /@ pointsOnSphere];
    Print["getPointsOnCYIPS: First L matrix: ", L[[1]]];
    (* Apply metric transformation to sphere points *)
    transformedSphere = Table[
        Table[
            Sum[L[[j]][[k, b]] pointsOnSphere[[j, k, a]], {b, Length[L[[j]]]}],
            {k, Length[pointsOnSphere[[j]]]}, {a, Length[pointsOnSphere[[j, 1]]]}
        ],
        {j, Length[pointsOnSphere]}
    ];
    Print["getPointsOnCYIPS: Sample transformed point: ", transformedSphere[[1, 1]]];
    Print["getPointsOnCYIPS: Sample original point: ", pointsOnSphere[[1, 1]]];
    Print["getPointsOnCYIPS: Are they different? ", transformedSphere[[1, 1]] != pointsOnSphere[[1, 1]]];
    subst = {};
    pts = {};
    For[j = 1, j <= Length[dimPs], j++,
        AppendTo[subst, 
            Table[varsUnflat[[j, a]] -> 
                Sum[params[[j, b]] transformedSphere[[j, b, a]], {b, Length[params[[j]]]}],
                {a, Length[varsUnflat[[j]]]}
            ]
        ];
    ];
    subst = Flatten[subst];
    
    res = FindInstance[
        Table[eqns[[i]] == 0, {i, Length[eqns]}] /. subst,
        Variables[Flatten[params]],
        Complexes,
        1000,
        WorkingPrecision -> precision
    ];
    
    pts = Chop[(varsUnflat /. subst) /. res];
    
    (* Go to patch where largest coordinate is 1 *)
    absPts = Abs[pts];
    For[i = 1, i <= Length[pts], i++,
        pts[[i]] = Chop[Flatten[
            Table[pts[[i, j]] / pts[[i, j, Ordering[absPts[[i, j]], -1][[1]]]], 
                {j, Length[dimPs]}]
        ]];
    ];
    
    Return[pts];
)];

(* Sample points and compute weights *)
SamplePointsIPS[varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_, 
    coefficients_, exponents_, L_, numPts_, dimCY_, \[Kappa]In_: 1, precision_: 20] := Module[
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
    
    (* Determine parameter distribution (same as original code) *)
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
    numPoints = Ceiling[numPts / 5];
    Clear[t];
    params = Table[
        Join[{1}, Table[Subscript[t, j, k], {k, numParamsInPn[[j]]}]],
        {j, Length[numParamsInPn]}
    ];
    
    pointsOnSphere = ParallelTable[
        SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
        {i, Length[dimPs]},
        DistributedContexts -> Automatic
    ];

    Print["SamplePointsIPS: About to sample ", numPoints, " intersections with L having dimensions: ", Dimensions /@ L];
    (* Sample points with metric transformation *)
    pts = ParallelTable[
        getPointsOnCYIPS[varsUnflat, numParamsInPn, dimPs, params,
            Table[pointsOnSphere[[i, p + (b - 1) numPoints]], 
                {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
            eqns, L, precision],
        {p, numPoints},
        DistributedContexts -> Automatic
    ];
    
    pts = Flatten[pts, 1];
    Print["SamplePointsIPS: Found ", Length[pts], " raw points"];
    
    (* Compute metric *)
    g = getFS[varsUnflat, bvarsUnflat, Table[ConjugateTranspose[L[[k]]] . L[[k]], {k, Length[L]}]];
    
    (* Compute weights *)
    allPts = ParallelTable[
        Module[{point, patch},
            point = pts[[i]];
            patch = getAbsMaxPos[point];
            {w, \[CapitalOmega]} = getWeightOmegas[varsFlat, bvarsFlat, varsUnflat, dimPs,
                g, eqns, beqns, point, Conjugate[point], patch, \[Kappa]In];
            Chop[{point, w, \[CapitalOmega], patch, L}]
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

    (* NEW DEBUG PRINTS HERE *)
    Print["SamplePointsIPS: First 3 points and their weights:"];
    Print["  Point 1: ", allPts[[1, 1]], " -> weight: ", allPts[[1, 2]]];
    If[Length[allPts] >= 2,
        Print["  Point 2: ", allPts[[2, 1]], " -> weight: ", allPts[[2, 2]]];
    ];
    If[Length[allPts] >= 3,
        Print["  Point 3: ", allPts[[3, 1]], " -> weight: ", allPts[[3, 2]]];
    ];
    Print["SamplePointsIPS: Weight statistics - Min: ", Min[allPts[[;;, 2]]], 
          ", Max: ", Max[allPts[[;;, 2]]], ", Mean: ", Mean[allPts[[;;, 2]]]];
    Print["SamplePointsIPS: L matrix being used: ", L[[1, 1;;2, 1;;2]]];
    
    Return[{allPts, \[Kappa]}];
)];

(* Main function for IPS on CICYs *)
GeneratePointsMCICYIPS[TotalNumPts_, NumRegions_, dimPs_, coefficients_, exponents_, 
    precision_: 20, verbose_: 0, frontEnd_: False] := Module[
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
        dimPs, coefficients, exponents, Ls[[-1]], NumPts, dimCY, 1, precision];
    \[Kappa]s = {\[Kappa]};
    PrintMsg["Calculated \[Kappa]=" <> ToString[\[Kappa]], frontEnd, verbose];
    
    (* Iteratively generate new regions *)
    For[r = 1, r <= Floor[NumRegions/2], r++,
        Ls = Join[Ls, GetNewLambdas[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, eqns, beqns, allPts, Ls, \[Kappa]s, dimCY]];
        
        PrintMsg["Processing region " <> ToString[2*r], frontEnd, verbose];
        {newPts, \[Kappa]} = SamplePointsIPS[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, coefficients, exponents, Ls[[-2]], NumPts, dimCY, \[Kappa], precision];
        allPts = Join[allPts, newPts];
        AppendTo[\[Kappa]s, \[Kappa]];
        
        PrintMsg["Processing region " <> ToString[2*r + 1], frontEnd, verbose];
        {newPts, \[Kappa]} = SamplePointsIPS[varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
            dimPs, coefficients, exponents, Ls[[-1]], NumPts, dimCY, \[Kappa], precision];
        allPts = Join[allPts, newPts];
        AppendTo[\[Kappa]s, \[Kappa]];
    ];
    
    PrintMsg["done.", frontEnd, verbose];
    
    Return[{allPts[[;;, 1]], allPts[[;;, 2]], allPts[[;;, 3]], \[Kappa]s, {dimCY}}];
)];