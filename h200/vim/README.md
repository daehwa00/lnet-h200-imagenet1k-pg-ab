# Vim-Tiny seed521 on H200

This is the same original Vim-Tiny used on qlab: 7,148,008 parameters,
middle class token, bidirectional-v2, no pretrained weights. Seed521 only,
100epochs, physical/effective batch256, BF16, AdamW LR0.003 and the existing
matched augmentation/schedule. Extra torch.compile stays disabled, matching
the faster qlab native-kernel path. GPU Mixup/yield-first/spawn loaders apply.

`H200_VIM_TINY_ONLY=1 bash h200/run_baselines.sh` selects this dedicated lane,
the hash-locked `requirements.lock`, and control/imagenet1k-vim-tiny-s521.
Use the official request form as daehwa00 with GPU allocation7 (whole H200),
and an exact H200_EXPECTED_COMMIT. No API secrets belong in the request.

The native preparation helper checks both official release-wheel SHA256s.
Only selective_scan_cuda is extracted from the Mamba wheel; the original
Vim source supplies its Python implementation, avoiding unrelated Mamba2
dependencies. Causal-conv1d is installed without dependency changes. A local
linker-name alias to the real CUDA driver is kept inside the run directory,
not installed in system paths. The adapter/source and native library hashes
are recorded in the training contract.

Before training, a separate 2-step real-data preflight evaluates all50,000
validation images. It must finish without nonfinite loss before production
starts; preflight weights are never used for the100-epoch run.

Preparation evidence: locked72-package environment installs and passes
dependency checks; official wheels download with expected digests; extracted
scan includes sm90/H200 code; original model builds with7,148,008parameters.
A local Blackwell batch16 BF16 train/backward/optimizer/eval smoke passed.
This is not an H200 throughput measurement; actual H200 preflight happens
after resource allocation. CPU entrypoint/identity/checksum/linker tests and
relay tests/dry-run pass. No existing relay run ID is removed.

qlab handoff: `qlab_vim_handoff.py` freezes only the old queue supervisor,
not its active seed501 worker. The old supervisor receives its root STOP
marker. After a strict100-epoch seed501 result AND worker exit, it is resumed
to terminate, and a new seed509-only queue starts from the original qlab
runtime198093a. Seed521 is absent from that new queue. Checkpoints and run IDs
for501/509 are preserved; the H200521 run has a distinct relay identity.
