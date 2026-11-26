#!/usr/bin/env bash
set -euo pipefail
# Read-only user homes still expose private files. Stop before loading PR code.
for forbidden in /public/home /public4/home /var/run/docker.sock /run/containerd/containerd.sock /var/run/secrets/kubernetes.io/serviceaccount/token; do
    if [[ -e "$forbidden" ]]; then echo "::error::PR pod unexpectedly exposes $forbidden"; exit 1; fi
done
