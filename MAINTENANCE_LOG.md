# Maintenance log

> This file is appended automatically by a scheduled maintenance task, and the
> commits that touch it carry `[skip ci]`. Every entry corresponds to one real
> change that passed the repository's own checks before it was pushed.
>
> The entries are generated rather than hand-written, and they record
> maintenance work only — they are not a measure of manual development effort.

## 2026-09-22 — add a first unit-test suite for the torch-free pipeline maths

- **Change**: new tests/ covering letterbox, preprocess, nms, multiclass_nms, decode and the IoU/sigmoid helpers (16 tests); adds pytest.ini and a conftest.py that puts the repo root on sys.path
- **Verification**: pytest: 16 passed; ruff: all checks passed

