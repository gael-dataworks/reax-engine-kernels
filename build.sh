#!/bin/bash
# Builds one engine CUDA library inside the pinned toolchain image (see Dockerfile.build).
#   docker run --rm --network none -v <so_sources>:/src:ro -v <outdir>:/out -e HOST_UID=$(id -u) [-e PIN=1] \
#     reax-so-build:13.2.86 bash /src/build.sh <lib>
# (rebuild_all.sh does this for all five libraries and checks the result.)
# Two build-environment inputs end up in the .so besides the sources and flags (see README):
#  - the source's absolute path: cudafe++ puts crc32(path) into internal symbol names, so each library is
#    compiled in the directory it was originally built in (dir= below);
#  - nvcc's process ID, which names its temp files; PIN=1 runs nvcc as the PID recorded in the shipped .so.
# Use a fresh container per library: PIDs are pinned by counting up from 1 in the container's PID namespace.
set -euo pipefail
lib=$1
case $lib in
  # lib)            build dir, nvcc pid, nvcc arguments (exactly as originally built)
  libreax_gv)        dir=/root/work/rv2/cuda; pid=0x88f9e; args=(-O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Icutlass/include -Xptxas -v -shared -Xcompiler -fPIC gateup_vis2.cu -o libreax_gv2.so); out=libreax_gv2.so ;;
  libreax_plain)     dir=/root/work/rv2/cuda; pid=0x89b85; args=(-O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Icutlass/include -Xptxas -v -shared -Xcompiler -fPIC gemm_plain2.cu -o libreax_plain2.so); out=libreax_plain2.so ;;
  libreax_fmha_post) dir=/root/work/rv2/cuda; pid=0x85026; args=(-O3 -arch=sm_120 -fmad=false -shared -Xcompiler -fPIC reax_fmha_v11.cu -o libreax_fmha_post.so); out=libreax_fmha_post.so ;;
  libreax_howu)      dir=/root/work/rv2/cuda; pid=0x87182; args=(-O3 -arch=sm_120 -fmad=false -DRPF=10 -shared -Xcompiler -fPIC reax_howu.cu reax_kkt.cu reax_howu_norm_v3.cu reax_rarn_final.cu reax_fmha_v8.cu reax_conv_v2.cu -o libreax_howu.so); out=libreax_howu.so ;;
  libreax_slices)    dir=/root/work/engine/csrc; pid=0x31aa; args=(-O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Icutlass/include -shared -Xcompiler -fPIC gemm_slices_v2.cu -o libreax_slices_v2.so); out=libreax_slices_v2.so ;;
  *) echo "unknown lib $lib" >&2; exit 2 ;;
esac
nvcc --version | tail -1
mkdir -p $dir && cd $dir
cp /src/$lib/*.cu .
ln -s /cutlass cutlass
if [ "${PIN:-0}" = 1 ]; then
  gcc -O2 -o /tmp/pidpin /src/pidpin.c
  /tmp/pidpin $pid nvcc "${args[@]}"
else
  nvcc "${args[@]}"
fi
cp $out /out/$lib.so
[ -n "${HOST_UID:-}" ] && chown $HOST_UID /out/$lib.so
sha256sum /out/$lib.so
