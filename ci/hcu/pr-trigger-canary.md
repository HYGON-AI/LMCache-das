<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# HCU PR trigger canary

This diagnostic-only PR checks the normal pull request event against current main.
It changes no runtime code, workflow, test assertion or model configuration.
The workflow must build the PR merge revision and report its actual failures.
This canary is not intended for merge.