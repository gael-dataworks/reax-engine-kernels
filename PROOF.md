# Clean rebuild record

A clean rebuild from this repository, run on a separate host from the original build machine:

- **When:** 2026-10-20 21:23 UTC.
- **Host:** x86_64, Ubuntu 24.04, Docker 29.2.1, 120 CPU threads; no GPU used.
- **Command:** `./rebuild_all.sh /root/audit/so_rebuilt` (PIN=1, the default).
- **Toolchain image:** `reax-so-build:13.2.86`, image ID `sha256:60f71ab28d101c87510c1c35626cf550c59d4cd79bc77ede8fcd970945cb149e`.
  It is built from `nvidia/cuda@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63`.

The build logs show each nvcc run under its recorded PID, with the resulting hashes:

```
libreax_gv.so         pidpin: running nvcc as pid 561054 (0x88f9e)  5a74f92518af5a42ff0b5490e6aa80df149ee8aba4863bc5f7d6f3889ff40715
libreax_slices.so     pidpin: running nvcc as pid 12714 (0x31aa)    0b4826bc21177e2f8b7e6eb9123329f4bcfcadc576c8c2cd702feeedc92b3890
libreax_howu.so       pidpin: running nvcc as pid 553346 (0x87182)  84174a5e1b79eb357565398862733fdcc249027b873ef30b69699ca2071c3739
libreax_fmha_post.so  pidpin: running nvcc as pid 544806 (0x85026)  7c2b6ba5b6412457b54b8f3b071416833f36f5d9812f838c2eecf40934a717c5
libreax_plain.so      pidpin: running nvcc as pid 564101 (0x89b85)  b3fb04ef31bfca221c21efca57ce908e232d80c187bf5e5649ef89f61b20cd30
```

`sha256sum -c SHIPPED_SHA256.txt` passed for all five:

```
libreax_gv.so: OK
libreax_slices.so: OK
libreax_howu.so: OK
libreax_fmha_post.so: OK
libreax_plain.so: OK
```

The hashes in `SHIPPED_SHA256.txt` are those of the files extracted from the committed image with
`scripts/extract_shipped.sh`:

`docker.io/kernelsmith1/sn92-engine@sha256:576fea84b347fc85160f788d737db616be4b5ca4cde19268e0ce1d1f5cdfe12f`
