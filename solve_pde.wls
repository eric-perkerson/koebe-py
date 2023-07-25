#!/usr/bin/env wolframscript

<< NDSolve`FEM`

ReadPolyFile[fileName_] :=
 Module[{txtFile, rawNode, lines, parsedNodeHeader, parsedNodeLines,
   parsedEdgeHeader, numEdges, parsedEdgeLines, parsedHoleHeader,
   numHoles, parsedHoleLines, edges, pointsInHoles, coordinates,
   bdryMarkers, numVertices}, txtFile = fileName <> ".txt";
  If[! FileExistsQ[txtFile],
   If[FileExistsQ[fileName], CopyFile[fileName, txtFile];,
    Print["File not found"]]];
  rawNode = Import[txtFile];
  lines = StringSplit[rawNode, "\n"];
  parsedNodeHeader = ToExpression@StringSplit[lines[[1]]];
  numVertices = parsedNodeHeader[[1]];
  parsedNodeLines =
   ToExpression@(StringSplit@lines[[2 ;; numVertices + 1]]);
  parsedEdgeHeader =
   ToExpression@StringSplit[lines[[numVertices + 2]]];
  numEdges = parsedEdgeHeader[[1]];
  parsedEdgeLines =
   ToExpression@(StringSplit@
      lines[[numVertices + 3 ;; numVertices + numEdges + 2]]);
  parsedHoleHeader =
   ToExpression@StringSplit[lines[[numVertices + numEdges + 3]]];
  numHoles = parsedHoleHeader[[1]];
  parsedHoleLines =
   ToExpression@(StringSplit@
      lines[[numVertices + numEdges + 4 ;;
         numVertices + numEdges + numHoles + 3]]);
  coordinates = parsedNodeLines[[;; , {2, 3}]];
  bdryMarkers = parsedNodeLines[[;; , 4]];
  edges = parsedEdgeLines[[;; , {2, 3}]];
  pointsInHoles = parsedHoleLines[[;; , {2, 3}]];
  Assert[numVertices == Length[coordinates]];
  DeleteFile[txtFile];
  Return[{coordinates, bdryMarkers, edges, pointsInHoles}]]

ReadNodeFile[fileName_] :=
 Module[{txtFile, rawNode, lines, parsedHeader, parsedLines,
   coordinates, bdryMarkers, numVertices},
  txtFile = fileName <> ".txt";
  If[! FileExistsQ[txtFile],
   If[FileExistsQ[fileName], CopyFile[fileName, txtFile];,
    Print["File not found"]]];
  rawNode = Import[txtFile];
  lines = StringSplit[rawNode, "\n"];
  parsedHeader = ToExpression@StringSplit[lines[[1]]];
  parsedLines = ToExpression@(StringSplit@lines[[2 ;;]]);
  coordinates = parsedLines[[;; , {2, 3}]];
  bdryMarkers = parsedLines[[;; , 4]];
  numVertices = parsedHeader[[1]];
  Assert[numVertices == Length[coordinates]];
  DeleteFile[txtFile];
  Return[{coordinates, bdryMarkers}]]

ReadEleFile[fileName_] :=
 Module[{txtFile, rawEle, lines, parsedHeader, parsedLines, triangles,
    numTriangles}, txtFile = fileName <> ".txt";
  If[! FileExistsQ[txtFile],
   If[FileExistsQ[fileName], CopyFile[fileName, txtFile];,
    Print["File not found"]]];
  rawEle = Import[txtFile];
  lines = StringSplit[rawEle, "\n"];
  parsedHeader = ToExpression@StringSplit[lines[[1]]];
  parsedLines = ToExpression@(StringSplit@lines[[2 ;;]]);
  triangles = parsedLines[[;; , {2, 3, 4}]];
  numTriangles = parsedHeader[[1]];
  Assert[numTriangles == Length[triangles]];
  DeleteFile[txtFile];
  Return[triangles]]

readData[fullFileStem_] :=
 Module[{regionCoordinates, regionBdryMarkers, regionEdges,
   pointsToDeviateForOmega0, pointsInHoles, coordinates, bdryMarkers,
   triangles, topology,
   omega0Seq},
  {regionCoordinates, regionBdryMarkers, regionEdges, pointsInHoles} = ReadPolyFile[fullFileStem <> ".poly"];
  {coordinates, bdryMarkers} = ReadNodeFile[fullFileStem <> ".node"];
  triangles = ReadEleFile[fullFileStem <> ".ele"];
  topology = ReadEleFile[fullFileStem <> ".topo.ele"];
  (*Modify pointsInHoles to include an extra copy of the last hole*)
  AppendTo[pointsInHoles, pointsInHoles[[-1]]];
  Return[{regionCoordinates, regionBdryMarkers, regionEdges,
    pointsInHoles, coordinates, bdryMarkers, triangles, topology}]]


makeRegionData[regionCoordinates_, regionEdges_, pointsInHoles_,
  epsilon_] :=
 Module[{componentEndingVertices, regionBdryComponents,
   exteriorVertices, interiorVerticesByHole, region, regionSkeleton,
   nHoles, nClasses, selectedInteriorBdryPoints,
   selectedExteriorBdryPoint, omega0SeqInterior, omega0SeqExterior,
   omega0Seq},
  componentEndingVertices =
   Flatten@Position[Differences@regionEdges[[;; , 2]], x_ /; x < 0] +
    1;
  PrependTo[componentEndingVertices, 0];
  regionBdryComponents =
   Table[
    Range[componentEndingVertices[[i]] + 1,
     componentEndingVertices[[i + 1]]], {i,
     Length[componentEndingVertices] - 1}];
  exteriorVertices = regionBdryComponents[[1]];
  interiorVerticesByHole = regionBdryComponents[[2 ;;]];
  region = If[nHoles == 1,
    BoundaryMeshRegion[regionVertices,
     Line[Append[exteriorVertices, exteriorVertices[[1]]]],
     Line[
      Append[interiorVerticesByHole[[1]],
       interiorVerticesByHole[[1, 1]]] ]],
    BoundaryMeshRegion[regionVertices,
     Line[Append[exteriorVertices, exteriorVertices[[1]]]],
     Line[
      Append[interiorVerticesByHole[[1]],
       interiorVerticesByHole[[1, 1]]] ],
     Line[
      Append[interiorVerticesByHole[[2]],
       interiorVerticesByHole[[2, 1]]] ]]
    ];
  regionSkeleton = {MeshCoordinates[region], MeshCells[region, 1]};
  nHoles = Length[interiorVerticesByHole];
  nClasses = nHoles + 1;
  selectedInteriorBdryPoints = interiorVerticesByHole[[;; , 1]];
  selectedExteriorBdryPoint = exteriorVertices[[1]];
  omega0SeqInterior =
   regionCoordinates[[selectedInteriorBdryPoints]] +
    Map[(epsilon*#/Norm[#]) &,
     regionCoordinates[[selectedInteriorBdryPoints]] -
      pointsInHoles[[;; -2]]];
  omega0SeqExterior =
   regionCoordinates[[selectedExteriorBdryPoint]] +
    Map[(epsilon*#/Norm[#]) &, -regionCoordinates[[
        selectedExteriorBdryPoint]] + pointsInHoles[[-1]]];
  omega0Seq = Join[omega0SeqInterior, {omega0SeqExterior}];
  Return[{exteriorVertices, interiorVerticesByHole, region,
    regionSkeleton, nHoles, nClasses, omega0Seq}]
  ]

InteriorAndBoundaryEdgesCompiled::usage =
  "InteriorAndBoundaryEdgesCompiled[{{triangleList,_Integer,2},{\
triangulationTopology,_Integer,2}}] runs the internals for \
InteriorAndBoundaryEdges.";
InteriorAndBoundaryEdgesCompiled =
  Compile[{{triangleList, _Integer,
     2}, {triangulationTopology, _Integer, 2}},
   Module[{nTriangles, table, counter, i, j},
    nTriangles = Length[triangleList];
    table = Table[0, {3*nTriangles}, {3}];
    counter = 0;
    For[i = 1, i <= nTriangles, i++,
     For[j = 1, j <= 2, j++,
      If[triangulationTopology[[i, j]] == 0, counter = counter + 1;
        table[[counter, 1]] = triangleList[[i, j]];
        table[[counter, 2]] = triangleList[[i, j + 1]];
        table[[counter, 3]] = 0;, counter = counter + 1;
        table[[counter, 1]] = triangleList[[i, j]];
        table[[counter, 2]] = triangleList[[i, j + 1]];
        table[[counter, 3]] = 1;];];
     j = 3;
     If[triangulationTopology[[i, j]] == 0, counter = counter + 1;
      table[[counter, 1]] = triangleList[[i, j]];
      table[[counter, 2]] = triangleList[[i, 1]];
      table[[counter, 3]] = 0;, counter = counter + 1;
      table[[counter, 1]] = triangleList[[i, j]];
      table[[counter, 2]] = triangleList[[i, 1]];
      table[[counter, 3]] = 1;];];
    Return[table]]];

InteriorAndBoundaryEdges::usage =
  "InteriorAndBoundaryEdges[triangleList_, triangulationTopology_] \
creates a list of interior edges and a list of boundary edges.";
InteriorAndBoundaryEdges[triangleList_, triangulationTopology_] :=
  Module[{interiorAndBoundaryEdges,
    interiorEdgesOrientedTriangulation,
    boundaryEdgesOrientedTriangulation},
   interiorAndBoundaryEdges =
    InteriorAndBoundaryEdgesCompiled[triangleList,
     triangulationTopology];
   interiorEdgesOrientedTriangulation =
    Pick[interiorAndBoundaryEdges[[;; , 1 ;; 2]],
     interiorAndBoundaryEdges[[;; , 3]] /. {0 -> False, 1 -> True}];
   boundaryEdgesOrientedTriangulation =
    Pick[interiorAndBoundaryEdges[[;; , 1 ;; 2]],
     interiorAndBoundaryEdges[[;; , 3]] /. {1 -> False, 0 -> True}];
   Return[{interiorEdgesOrientedTriangulation,
     boundaryEdgesOrientedTriangulation}]];

EdgesWrapCompiled::usage =
  "Edges[{{polygon,_Integer,1}}] takes a list of indices and returns \
the list of pairs from one index to the next.";
EdgesWrapCompiled =
  Compile[{{polygon, _Integer, 1}},
   Append[Partition[polygon, 2, 1], {polygon[[-1]], polygon[[1]]}]];


WritePdeFile[fileName_, pdeValues_] :=
 Block[{nHoles, vertices, numVert, dimensions, numAttr,
   numBoundaryMarkers, edges, numEdges, stream, i, nValues},
  nValues = Length[pdeValues];
  stream = OpenWrite[fileName <> ".pde"];
  WriteString[stream,
   ToString[nValues] <> " " <> "\n"
   ];
  For[i = 1, i <= nValues, i++,
   WriteString[stream,
     ToString[i] <> " " <> ToString[CForm[pdeValues[[i]] ] ] <> "\n" ];
   ];
  Close[stream];
  ]


(*Begin Main*)
(* argv=Rest@$ScriptCommandLine; *)

argv = {"test_example_4"};
If[Length[argv] == 0, Print["No file stem given, exiting"];
 Exit[]]
fileStem = argv[[1]];
fullFileStem =
  FileNameJoin[{"/Users/eric/Code/planar-domains/regions/",
    fileStem, fileStem}];

{regionVertices, regionBdryMarkers, regionEdges, pointsInHoles,
   coordinates, bdryMarkers, triangles, triangulationTopology} =
  readData[fullFileStem];

{exteriorVertices, interiorVerticesByHole, region, regionSkeleton, nHoles, nClasses, omega0Seq} =
makeRegionData[regionVertices, regionEdges, pointsInHoles, 50];

T = {coordinates,
   Partition[Flatten[EdgesWrapCompiled /@ triangles], 2], triangles};
T[[2]] = Union[Sort /@ T[[2]]];

{interiorEdgesOrientedTriangulation,
   boundaryEdgesOrientedTriangulation} =
  InteriorAndBoundaryEdges[T[[3]], triangulationTopology];
eMesh = ToElementMesh["Coordinates" -> T[[1]],
   "MeshElements" -> {TriangleElement[T[[3]]]}];
exteriorBoundary =
  Line[regionVertices[[#]] & /@
    Append[exteriorVertices, exteriorVertices[[1]]]];
eMesh = ToElementMesh["Coordinates" -> T[[1]],
   "MeshElements" -> {TriangleElement[T[[3]]]}];
exteriorBoundary =
  Line[regionVertices[[#]] & /@
    Append[exteriorVertices, exteriorVertices[[1]]]];

interiorBoundaries =
  Table[Line[
    regionVertices[[#]] & /@
     Append[interiorVerticesByHole[[i]],
      interiorVerticesByHole[[i, 1]]]], {i, nHoles}];
\[CapitalGamma]1 =
  DirichletCondition[u[x, y] == 1,
   RegionDistance[exteriorBoundary, {x, y}] < 1*10^(-5)];
\[CapitalGamma]2 =
  DirichletCondition[u[x, y] == 0,
   Min[RegionDistance[#, {x, y}] & /@ interiorBoundaries] < 1*10^(-5)];

Print["Solving the PDE..."]

f = NDSolveValue[{-Laplacian[u[x, y], {x, y}] ==
     0, \[CapitalGamma]1, \[CapitalGamma]2},
   u, {x, y} \[Element] eMesh];
fValues = f @@@ T[[1]];

WritePdeFile[
 FileNameJoin[{"/Users/eric/Code/planar-domains/regions/",
   fileStem, fileStem}], fValues]

Print["Wrote the PDE solution to file: " <> FileNameJoin[{"/Users/eric/Code/planar-domains/regions/",
   fileStem, fileStem}]]
