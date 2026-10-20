#!/bin/bash
# Clean rebuild of the five engine CUDA libraries from source, then a byte-for-byte check against the shipped files.
#   ./rebuild_all.sh [outdir]     needs docker (network only for the image build), no GPU
#   PIN=0 ./rebuild_all.sh ...    skip PID pinning (see README: only nvcc's temp-file name then differs)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath -m "${1:-$here/rebuilt}")
mkdir -p "$out"
docker build ${NO_CACHE:+--no-cache} -f "$here/Dockerfile.build" -t reax-so-build:13.2.86 "$here"
pids=()
for lib in libreax_gv libreax_slices libreax_howu libreax_fmha_post libreax_plain; do
  docker run --rm --network none -v "$here":/src:ro -v "$out":/out -e HOST_UID="$(id -u)" -e PIN="${PIN:-1}" \
    reax-so-build:13.2.86 bash /src/build.sh $lib > "$out/$lib.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "a build failed; see $out/*.log" >&2; done
cd "$out" && sha256sum -c "$here/SHIPPED_SHA256.txt"
