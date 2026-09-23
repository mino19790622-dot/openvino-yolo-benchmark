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

## 2026-09-23 — note that Val2017Set.images() skips unreadable images

- **Change**: coco_eval_set.py: add a docstring to Val2017Set.images() recording that images are downloaded lazily and unreadable ones are skipped silently, so the number yielded can be lower than n() while n_gt() still counts every id
- **Verification**: pytest: 16 passed; ruff: all checks passed

