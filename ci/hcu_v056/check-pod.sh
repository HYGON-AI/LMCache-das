#!/usr/bin/env bash
set -euo pipefail
# Read-only user homes still expose private files. Stop before loading PR code.
for forbidden in /public/home /public4/home /var/run/docker.sock /run/containerd/containerd.sock /var/run/secrets/kubernetes.io/serviceaccount/token; do
    if [[ -e "$forbidden" ]]; then echo "::error::PR pod unexpectedly exposes $forbidden"; exit 1; fi
done
if [[ -d /ci_public ]]; then
    command -v findmnt >/dev/null || { echo "::error::Cannot inspect /ci_public mounts"; exit 1; }
    if opts="$(findmnt -rn -M /ci_public -o VFS-OPTIONS)"; then
        [[ ",$opts," == *,ro,* ]] || { echo "::error::PR pod exposes a writable /ci_public mount"; exit 1; }
    fi
fi
