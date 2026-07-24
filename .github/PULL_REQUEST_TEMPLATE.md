<!-- Thanks for contributing! Keep PRs focused on one logical change. -->

## Summary

<!-- What does this change and why? For a kernel change, state the memory-traffic
or roofline reasoning, not just the diff. -->

## Type of change

- [ ] Bug fix (incorrect results / crash)
- [ ] Performance (faster, same results)
- [ ] New kernel or utility
- [ ] Documentation
- [ ] Tests
- [ ] Chore / tooling

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `CUDA_VISIBLE_DEVICES="" pytest -q` passes (CPU subset; GPU tests skip)
- [ ] GPU correctness tests pass on a real GPU, or N/A (CPU-only change)
- [ ] New public symbols are exported from `triton_kernels/__init__.py`
- [ ] Behavior changes are reflected in `CHANGELOG.md` and, if relevant, `docs/`

## Validation

<!-- How did you verify correctness? Which reference did you compare against, on
which GPU (or CPU for reference math)? Paste the relevant test output. -->
