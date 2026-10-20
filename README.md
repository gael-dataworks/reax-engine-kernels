# Engine CUDA libraries: sources and reproducible build

This repository rebuilds the five CUDA libraries shipped in UID 198's REAX engine-track entry byte for byte from
source. That entry is `docker.io/kernelsmith1/sn92-engine@sha256:576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f`
(tag `x16`), and the libraries are in `/opt/reax/engine_pkg/`.
`scripts/extract_shipped.sh` extracts them from the image and checks them against `SHIPPED_SHA256.txt`.

```
./rebuild_all.sh            # builds the toolchain image, compiles each library in a fresh container,
                            # then runs: sha256sum -c SHIPPED_SHA256.txt  -> all five "OK"
```

Requirements: an x86_64 Linux host with Docker and about 12 GB of disk. No GPU is needed. Network access is
needed only for the image build; the compile containers run with `--network none`. A run takes about 5 minutes.

| Library | Sources | Compiled in | nvcc flags | nvcc PID |
|---|---|---|---|---|
| libreax_gv.so | `libreax_gv/gateup_vis2.cu` | `/root/work/rv2/cuda` | `-O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Icutlass/include -Xptxas -v -shared -Xcompiler -fPIC` | 0x88f9e |
| libreax_plain.so | `libreax_plain/gemm_plain2.cu` | `/root/work/rv2/cuda` | same as gv | 0x89b85 |
| libreax_slices.so | `libreax_slices/gemm_slices_v2.cu` | `/root/work/engine/csrc` | same as gv, without `-Xptxas -v` | 0x31aa |
| libreax_howu.so | `libreax_howu/reax_howu.cu reax_kkt.cu reax_howu_norm_v3.cu reax_rarn_final.cu reax_fmha_v8.cu reax_conv_v2.cu` | `/root/work/rv2/cuda` | `-O3 -arch=sm_120 -fmad=false -DRPF=10 -shared -Xcompiler -fPIC` | 0x87182 |
| libreax_fmha_post.so | `libreax_fmha_post/reax_fmha_v11.cu` | `/root/work/rv2/cuda` | `-O3 -arch=sm_120 -fmad=false -shared -Xcompiler -fPIC` | 0x85026 |

`build.sh` holds the exact command lines. gv, plain and slices were written as `libreax_gv2.so`,
`libreax_plain2.so` and `libreax_slices_v2.so` and renamed when packaged. The output name is not recorded in the
library, because no soname is set.

## Toolchain (`Dockerfile.build`)

- **Base image:** `nvidia/cuda:13.2.1-devel-ubuntu24.04`, pinned by digest.
- **CUDA packages:** `cuda-nvcc`, `cuda-crt`, `cuda-cudart` and `cuda-cudart-dev` 13.2.86-1, installed over the
  image's 13.2.78. The shipped libraries carry the string `Cuda compilation tools, release 13.2, V13.2.86`.
- **Host compiler and linker:** GCC `13.3.0-6ubuntu2~24.04.1` and binutils `2.42-4ubuntu2.10`, both from the
  base image. The `.comment` section of each library records the GCC version.
- **CUTLASS:** v4.2.1, commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, at `/cutlass`. gv, plain and slices
  use its headers.

## Build inputs that reach the binary besides sources and flags

Two more inputs end up in the output. Both can be checked against the shipped files directly.

1. **The absolute source path.** cudafe++ names internal-linkage entities `_INTERNAL_<crc32 of the source
   path>_<len>_<file>_cu_…`, so gv, plain and slices are compiled at their original paths. The CRC32 of each path
   equals the value in the shipped library:

   ```
   strings -a libreax_gv.so | grep -o '_INTERNAL_[0-9a-f]*_14_gateup_vis2_cu' | sort -u    # _INTERNAL_cd7cfbfd_...
   python3 -c 'import zlib; print("%08x" % zlib.crc32(b"/root/work/rv2/cuda/gateup_vis2.cu"))'      # cd7cfbfd
   ```

   The other values are `/root/work/rv2/cuda/gemm_plain2.cu` → `4f8391b7` and
   `/root/work/engine/csrc/gemm_slices_v2.cu` → `92fc1016`. howu and fmha_post have no such names, so their
   build path does not matter.
2. **nvcc's process ID.** nvcc names its intermediate files `/tmp/tmpxft_<pid>_00000000-<n>_<source>.*`. The host
   linker records the generated C++ file's name as a FILE symbol in `.symtab`, which is not loaded at runtime and
   not covered by the build-id. Compare `strings -a libreax_gv.so | grep tmpxft` with `tmpxft_00088f9e_…`.
   `pidpin.c` runs nvcc as that PID in a fresh container: it forks and reaps throwaway children until the
   namespace's PID counter reaches the value, then starts nvcc. It does not touch the output.

Without pinning (`PIN=0 ./rebuild_all.sh`), the rebuilt libraries differ from the shipped ones only in those
8-hex-digit `tmpxft` strings in `.strtab`. `tools/elfdiff.py rebuilt.so shipped.so` lists the differing bytes per
section.

The ptxas command line is recorded in the fatbinary as well, for example `-v  -arch sm_120 -m 64` with
`-Xptxas -v`. That is how the slices flags were pinned down: with `-Xptxas -v`, its fatbinary differs.

## Source of libreax_slices.so

The `gemm_slices_v2.cu` used for the shipped build was not kept, because the box it was built on is gone.
`libreax_slices/gemm_slices_v2.cu` is `engine/csrc/gemm_slices.cu` with the six TBK=64 configurations (16–21)
removed; those were added to that file after `libreax_slices.so` was built. In the resulting file:

- `SLICE_CFGS` ends at configuration 15.
- `reax_slices_ncfg()` returns 16.
- The name table and the range check have 16 entries.

Compiling this file reproduces the shipped `libreax_slices.so` byte for byte.

## Files

- `libreax_*/`: sources, one directory per library.
- `Dockerfile.build`: the toolchain image.
- `build.sh <lib>`: builds one library inside the image.
- `pidpin.c`: the PID helper described above.
- `rebuild_all.sh`: builds all five libraries and runs the check.
- `SHIPPED_SHA256.txt`: hashes of the five libraries as found in the x16 image.
- `tools/elfdiff.py`: per-section byte diff of two ELF files.
- `PROOF.md`: the recorded clean-rebuild run.
- `scripts/extract_shipped.sh`: extracts the shipped libraries from the image and checks their hashes.
- `LICENSE`: Apache License 2.0.

## Licence

The sources in this repository are licensed under the Apache License 2.0 (`LICENSE`). CUTLASS, which the build
fetches as headers, is under its own BSD-3-Clause licence.
