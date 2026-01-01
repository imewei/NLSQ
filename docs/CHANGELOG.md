# Documentation Changelog

## Unreleased
- Removed legacy StreamingOptimizer/StreamingConfig and the Adam warmup path from docs; use AdaptiveHybridStreamingOptimizer with HybridStreamingConfig (L-BFGS warmup) for large datasets.
- Removed DataGenerator, create_hdf5_dataset, and fit_unlimited_data from examples/docs; use LargeDatasetFitter or AdaptiveHybridStreamingOptimizer instead.
- Updated streaming workflow guidance to reflect hybrid streaming only.
