#!/bin/bash
# Extract the five kernel libraries from the committed engine image and check them against SHIPPED_SHA256.txt.
#   bash scripts/extract_shipped.sh [image@sha256:...]
set -euo pipefail
IMG=${1:-docker.io/kernelsmith1/sn92-engine@sha256:576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f}
OUT=shipped; mkdir -p "$OUT"
docker pull -q "$IMG" > /dev/null
C=$(docker create "$IMG")
trap 'docker rm "$C" > /dev/null' EXIT
for f in libreax_fmha_post.so libreax_gv.so libreax_howu.so libreax_plain.so libreax_slices.so; do
  docker cp -q "$C:/opt/reax/engine_pkg/$f" "$OUT/"
done
(cd "$OUT" && sha256sum -c ../SHIPPED_SHA256.txt)
