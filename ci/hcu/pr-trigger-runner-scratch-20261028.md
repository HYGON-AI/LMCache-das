# PR22 runner-local scratch validation

This diagnostic PR triggers one normal HCU PR CI run after PR22 was merged.
It is not intended for merge.

- Base/controller expected: `dc99710363826c1f1f738e01fa5dc4470dde334b`.
- Build one wheel from this PR's actual merge SHA.
- Validate full pytest inventory, active JUnit results, and approved exclusions.
- Run the existing Qwen LocalDisk and POSIX workloads without changes.
- Require successful framework, both model jobs, and hcu-tests results.
- Use runner-local scratch; PR report archiving remains disabled.
- Verify the small wheel bundle provenance/checksums and scoped cleanup.

Confirm controller identity from actual job inputs/checkout, not PR metadata alone.
The recurring monitor stays paused; this is a single manually authorized run.
