#!/usr/bin/env bash
# Run once as the asset maintainer on a node with writable /ci_public (never in PR CI).
set -euo pipefail
dir=/ci_public/lmcache-das/assets/upstream
name=lmcache-0.5.6-3e11e108.tar.gz
sha=6cbc61020b30609edec743a6c81c0a7ed91f1b5c081bbc513d75d0a01ee5b75e
verify() { [[ "$(sha256sum -- "$1" | cut -d ' ' -f 1)" == "$sha" ]]; }
[[ $# -eq 1 && -f "$1" && -d /ci_public ]] || { echo "Usage: bash stage-upstream.sh <downloaded archive> (requires /ci_public)" >&2; exit 2; }
mountpoint -q /ci_public || { echo "Expected /ci_public to be a mounted filesystem" >&2; exit 2; }
mkdir -p -- "$dir"
if [[ -e "$dir/$name" ]]; then verify "$dir/$name" || { echo "Conflicting existing archive; refusing to replace" >&2; exit 1; }; exit 0; fi
tmp="$(mktemp "$dir/.lmcache-v056.XXXXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
cp -- "$1" "$tmp"
verify "$tmp" || { echo "Wrong upstream checksum; not published" >&2; exit 1; }
chmod 0444 "$tmp"
ln -T -- "$tmp" "$dir/$name" || { verify "$dir/$name" || exit 1; }
sha256sum -- "$dir/$name"
