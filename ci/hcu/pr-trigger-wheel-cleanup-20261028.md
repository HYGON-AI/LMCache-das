# Post-PR24 wheel cleanup validation

One normal PR CI run requested by the maintainer after merging PR24.
This diagnostic PR is not intended for merge.

- Expected trusted controller: `c4af4fa0f1551f8156bf46c88359c5f98def8d19`.
- Confirm the actual job inputs and controller checkout, not only PR metadata.
- Build one wheel from the actual tested merge SHA.
- Validate full pytest inventory/JUnit and the unchanged approved exclusions.
- Exercise framework wheel publication and cleanup on the real runner host.
- Run the unchanged Qwen LocalDisk and POSIX workloads using that wheel.
- Require successful framework/model jobs and hcu-tests, with scoped cleanup.

PR report archiving stays disabled. The recurring monitor stays paused.
