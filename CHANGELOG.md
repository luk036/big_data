# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-20

### Features
- **`CountMinSketch.from_error`**: Added a factory that sizes width and depth from the standard `(epsilon, delta)` guarantee, and promoted `CountMinSketch` to the package root. (#f89e6e7, #b0255cb)
- **Bias-reduced CMS estimators**: Added `query_mean_min()`, conservative-update mode, `estimate_with_error()`/`additive_error_bound()`, merge/reset support, and total stream-weight tracking. (#b0255cb)

### Bug Fixes
- **HyperLogLog hashing and rank**: Unified every HyperLogLog on one 64-bit MurmurHash3/FNV family and fixed the rank to `leading zeros + 1`; the old formula only looked correct at small cardinalities because the linear-counting correction masked it. (#34b7002)
- **GPU HyperLogLog correctness**: Zero-initialise the register buffer (dirty pooled memory corrupted estimates), remove no-op expressions, and vectorise batch string hashing. (#34b7002)

### Performance
- **Shared HyperLogLog core**: `HyperLogLog`, `EDAHyperLogLog`, `CPUHyperLogLog` and `GPUHyperLogLog` now delegate to one estimator core, with a fused hash-and-fold device kernel and split H2D/kernel/D2H benchmark reporting. (#32c66de, #3d172ff, #34b7002)

### Testing & Code Quality
- **Switching-analyzer characterization tests**: Pinned activity/power, temporal, pattern and corner behaviour and the single-source layout. (#2f1cb16)
- **Switching-analysis orchestrator**: Replaced the ~1600-line duplicated demo monolith with a thin runner plus a dedicated `SwitchingPowerAnalyzer` module, and dropped dead imports and stale `type: ignore`s. (#4d324a1, #26b54e2)

### Documentation
- **bigdata4eda paper and deck**: Added the IEEEtran paper on Count-Min Sketch / HyperLogLog for EDA analytics and its Beamer deck. (#f386d12)

## [0.2.1] - 2026-10-09

### Fixed
- **Reproducible EDA suite**: Seeded the EDA demo RNG so the application-suite metrics are stable across runs. (#f0a0db3)
- **Consistent memory labels**: The switching-analysis demo's memory/accuracy table now uses the same float32 labels as the standalone module (1.2 / 7.8 / 39 / 234 KB). (#5dc159b)

## [0.2.0] - 2026-10-09

### Fixed
- **EDA application suite now runs**: Added the missing `EDAHyperLogLog._estimate_cardinality()` and replaced the salted built-in `hash()` with SHA-256 for correct, reproducible estimates. (#cd5f418)
- **EDA suite counts distinct patterns**: The seven applications now feed pattern signatures to the sketch instead of unique per-instance IDs, so estimates match ground truth (e.g. fanout 49/49, placement 0.3% error). (#964c0e6)
- **Count-Min Sketch merge correctness**: Hash functions are derived deterministically from `(width, depth)`, so a re-created sketch reuses the same hashes and `merge()` returns correct estimates. (#9312f61)
- **Reproducible sketch results**: Count-Min Sketch no longer seeds its hashes from the unseeded `random` module; demo runs are deterministic. (#c3ea6f9)
- **HyperLogLog demo ground truth**: The demo compares against the true distinct count (7000) rather than the stream length, removing the bogus ~30% error. (#183ff20)
- **Memory/accuracy labels**: Configuration labels now match the printed float32 memory figures (1.2 / 7.8 / 39 / 234 KB). (#c3ea6f9)

### Maintenance
- Ignore demo-generated figures at the repository root. (#3227f28)

## [0.1.0] - 2026-10-09

### Features
- **Count-Min Sketch switching analysis**: Packaged the Count-Min Sketch experiments for switching-activity tracking, including memory/accuracy trade-off, pattern-dependent, temporal and end-to-end switching-power demos. (#4a3cd3e)
- **HyperLogLog cardinality estimation**: Added CPU HyperLogLog and its EDA application suite. (#7cb933a)
- **GPU acceleration demos**: Added Numba-CUDA HyperLogLog variants (minimal and tuned with bias correction) and an interactive GPU Mandelbrot renderer. (#32ce446)

### Testing
- **Smoke tests**: Added module-import coverage plus correctness checks for Count-Min Sketch and HyperLogLog. (#7bad5a2)

### Documentation
- **README, license and metadata**: Added MIT license, README with module catalogue, and CHANGELOG/AUTHORS/CONTRIBUTING. (#7ca8acc)
- **Notebooks and figures**: Added Jupyter versions of the experiments and the generated figures. (#6153b72)

### Build & CI
- **Project setup**: Added `pyproject.toml`/`setup.cfg` (setuptools-scm), `tox`, pre-commit hooks and a GitHub Actions test matrix (Python 3.10-3.13). (#0732e44)
