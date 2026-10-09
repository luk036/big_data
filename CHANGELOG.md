# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
