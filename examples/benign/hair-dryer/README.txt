Hair dryer: proposed benign sample
=================================

A household hair dryer exterior with a concentrator nozzle, five rear intake
bars, two handle controls and a short cord. The OBJ contains geometry only;
the colors in preview.png are illustrative. Coordinates use arbitrary units.

Proposed label: 0 (benign), pending maintainer review.
Publication license: not yet selected; to be agreed before inclusion.

Files
-----
hair-dryer.obj       Triangle mesh.
preview.png         Front, rear and monochrome views of the same mesh.
case.json           Sample identity, proposed label, units and mesh checksum.
mesh-check.json     Checks on the saved mesh and repository reader compatibility.
proxy-results.json  Exact local scores, fixed thresholds and checkpoint hashes.
SHA256SUMS.txt      Checksums for the package contents.

Mesh checks
-----------
The mesh has 5,999 vertices and 12,014 triangles in one connected component.
Boundary-edge, manifoldness, orientation and duplicate-face checks pass. The
repository reader and sanitizer preserve the mesh. The five intake bars account
for its genus of five. Separate triangle self-intersection testing was not done.
mesh-check.json covers geometry; inference results are in proxy-results.json.

Local proxy check
-----------------
Six existing PointNet checkpoints classified this sample as positive; six
spectral checkpoints classified it as benign. All calls succeeded and same-seed
repetitions retained their classifications. Weights and validation thresholds
were fixed before inference; the OBJ was unchanged throughout the test.

The checkpoints were trained on public proxy shapes and ellipsoid negatives.
PointNet used three epochs, 128 sampled points and one network per checkpoint.
This is one exploratory case with overlapping training sets, so the counts do
not estimate a population false-positive rate or real-data performance.

Model and validation scripts, detailed study records and local checkpoints are
retained separately. This compact package provides the sample and recorded
evidence; it does not include the training data or weights needed to repeat
inference. All pose, scale and remeshing variants must keep the same template ID
and remain in one dataset split.
