Regularized complex second-moment normalization

Isolated Rust/Kernite candidate on the selected squared-imaginary score,
phase-mHC and LeRoPE main. Exact definition and frozen protocol are in
../COMPLEX_COVARIANCE_PLAN.txt. This uses uncentered token-local moments,
shrinkage rho=1/16 and epsilon=2^-23; no parameters or initialization changes.
The real mHC controller's RMS is retained. V is not normalized.

Build/test offline with features tf32-backward,fused-attention,tiled-transpose.
CUDA tests are ignored by default; run covariance_tests explicitly with
--ignored --test-threads=1. COVARIANCE_VALIDATION_OUTPUT names a fresh evidence
directory. fp32-affine-reference is diagnostic only. No Python execution.

Scientific training remains unstarted until numerical validation, independent
registration and frozen binary/source/PTX provenance. Existing numerical
limitations of all parent models remain; no convergence guarantee is claimed.
