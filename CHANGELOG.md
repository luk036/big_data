# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial packaging of the Count-Min Sketch / HyperLogLog experiment modules
  under the `big_data` package (`src` layout).
- Project scaffolding: `pyproject.toml`, `setup.cfg`, `tox.ini`, pre-commit
  hooks and a GitHub Actions test matrix.
- Smoke tests for module imports and the core sketch implementations.
