# Read-only K96 checkpoint probe

Submit as `daehwa00`, preferably using the same H200 storage pool as issue379.
One MIG slice is sufficient; the script does not use the GPU, train, download,
install dependencies, or write experiment files. Run with `python3 -B -u`.

The three bounded candidate directories refer only to the original K96
ImageNet-1K seed501 run. Reports distinguish result.json from actual weights.
If weights exist, output includes file size, SHA256 and a CPU-only metadata
inspection (90-second subprocess timeout; needs the image's existing torch).
It checks completed100epochs, expected contract,3,253,224parameters and the
1000-class classifier shape. Saved epoch100 validation accuracy is compared
with result.json when both are present. This is NOT a fresh ImageNet evaluation.

Historical Top1=72.374% is labeled historical and never presented as a newly
observed result when no file is visible. Missing files do not prove deletion.
Checkpoint deserialization is restricted to the original user-owned paths.
If torch/dependencies are unavailable, existence/hash/results are still reported
and checkpoint metadata inspection fails explicitly rather than installing code.
