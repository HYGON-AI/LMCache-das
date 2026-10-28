# Post-PR26 data-disk cache validation

This diagnostic-only PR starts from main
`39ee5c23e755a838fe66edb142a0a098f4643571` after PR26 was merged.
Do not merge this canary. It exists to create a fresh pull_request_target
opened event with the new trusted controller, not to reopen an older PR.

Run the normal framework/current-SHA wheel and full active pytest scope,
followed by the unchanged Qwen LocalDisk and POSIX jobs. Verify the actual
controller checkout, four run-scoped cache volumes on `/data`, capacity/direct
I/O preflight, model results/JUnit, required summary, and exact resource cleanup.
No weekly/accuracy workload, manual GPU job or recurring monitor is requested.
