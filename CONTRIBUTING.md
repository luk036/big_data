# Contributing

Contributions are welcome. To set up a development environment:

```bash
pip install -e .[testing]
```

Before submitting a change, please make sure the test suite passes:

```bash
pytest
```

and that the source tree is formatted:

```bash
pre-commit run --all-files
```

Please keep each experiment module self-contained and update `CHANGELOG.md` for
user-visible changes.
