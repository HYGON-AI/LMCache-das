# Current-main PR trigger canary

This temporary diagnostic change verifies the normal HCU PR workflow after
the repository patch-flow update was merged into main at
`9d91d5aa7d34fe9acb9fad8a1358b01c744e09e1`.

The actual job inputs and controller checkout must use that base commit.
The patch preparation log must show the repository CLI flow, not the
suspended strict source-patch gate.

This PR changes no business code, workflow logic, test assertions or model
parameters. It is for CI validation and is not intended to be merged.
