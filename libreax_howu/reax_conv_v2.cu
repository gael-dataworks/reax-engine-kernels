// v2 (branch-free exact SiLU division). Exact CUDA port of engine.conv_l2._conv_silu_l2_pc (Triton, sm_120 build): per (row, 128-channel unit) the 4-tap causal
// conv (fp32 fma chain over taps 0..3, skipped taps keep acc), bf16 rounding, SiLU = y / (1 + exp(-y)) with libdevice's
// ftz expf whose final scale multiply is fused with the +1, bf16 rounding; q/k units then L2-normalize over the unit:
// per thread (8 contiguous channels) sum = e1^2 then fma e0, e2..e7, half-warp butterfly xor 8,4,2,1, + eps,
// sqrt.approx.ftz, div.full 1/x, x * rstd -> bf16. Lane l of a half-warp owns channels 8*(l%16)..+7 (Triton's layout).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float d; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float fma_rm_ftz(float a, float b, float c) { float d; asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float sat_ftz(float a) { float d; asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float add_rn(float a, float b) { float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float sub_rn(float a, float b) { float d; asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float mul_rn(float a, float b) { float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float div_rn(float a, float b) { float d; asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float div_full(float a, float b) { float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float sqrt_ftz(float a) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float fma_bf(unsigned short a, unsigned short b, float c) { float d; asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(c)); return d; }
__device__ __forceinline__ unsigned short h16(uint32_t x, int hi) { return (unsigned short)(hi ? (x >> 16) : (x & 0xffffu)); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float blo(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float bhi(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

// 1 + exp(-y) exactly as Triton/libdevice computes it (the final 2^k scale fused with the +1)
__device__ __forceinline__ float one_plus_exp_neg(float y) {
  const float a = sub_rn(0.f, y);
  float t = fma_ftz(a, __uint_as_float(0x3BBB989Du), 0.5f);
  t = sat_ftz(t);
  const float j = fma_rm_ftz(t, 252.0f, __uint_as_float(0x4B400001u));
  const float jj = add_rn(j, __uint_as_float(0xCB40007Fu));
  float f = fma_ftz(a, __uint_as_float(0x3FB8AA3Bu), -jj);
  f = fma_ftz(a, __uint_as_float(0x32A57060u), f);
  const float sc = __uint_as_float(__float_as_uint(j) << 23);
  const float e = ex2_ftz(f);
  return fma_rn(e, sc, 1.0f);
}

__device__ __forceinline__ float rcp_approx(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
// y / (1 + exp(-y)) without div.rn's slow-path branch: refined reciprocal + one remainder correction, +-0 by select.
// Exhaustive check over all 65536 bf16 y (rv2/cuda/siluc_all_test.py): f32-bitwise equal to div_rn(y, one_plus_exp_neg(y))
// for every finite y > -87; others are flagged and recomputed with div_rn.
__device__ __forceinline__ float silu_c_fast(float y, bool& fl) {
  const float d = one_plus_exp_neg(y);
  float r = rcp_approx(d);
  r = fma_rn(fma_rn(-d, r, 1.f), r, r);
  float q = mul_rn(y, r);
  const float rem = fma_rn(-d, q, y);
  q = fma_rn(rem, r, q);
  q = y == 0.f ? y : q;
  fl |= !(y > -87.0f && fabsf(y) < INFINITY);
  return q;
}

template <int REP, bool SKIP_V, int NR>
__global__ void __launch_bounds__(128)
conv_l2_kernel(const __nv_bfloat16* __restrict__ qkv, long long SX, const __nv_bfloat16* __restrict__ cache, int C,
               const int* __restrict__ meta, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ q,
               __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, int Tf, float eps, int KD, int VD, int u0) {
  // CTA = 8 half-warps x NR rows each (rows tbeg + hw + 8 r); all tap loads of a thread's NR rows are issued together
  const int lane = threadIdx.x & 31, hw = threadIdx.x >> 4, l16 = lane & 15;
  const int u = u0 + blockIdx.y;
  const int c0 = u * 128 + 8 * l16;
  const int nqk = 2 * KD / 128;
  const int tbeg = blockIdx.x * (8 * NR);
  int4 m[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    const int t = tbeg + hw + 8 * r;
    m[r] = t < Tf ? *reinterpret_cast<const int4*>(meta + (long long)t * 4) : make_int4(-100, 0, 0, 0);
  }
  unsigned short wt[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint2 wv = *reinterpret_cast<const uint2*>(w + (long long)(c0 + i) * 4);
    wt[i][0] = h16(wv.x, 0); wt[i][1] = h16(wv.x, 1); wt[i][2] = h16(wv.y, 0); wt[i][3] = h16(wv.y, 1);
  }
  uint4 xv[NR][4];
#pragma unroll
  for (int r = 0; r < NR; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int src = m[r].x - 3 + j;
      const bool ok = src >= 0;
      const __nv_bfloat16* rp = src >= m[r].y ? qkv + (long long)(m[r].z + src) * SX : cache + (long long)(m[r].w + (ok ? src : 0)) * C;
      xv[r][j] = ok ? __ldg(reinterpret_cast<const uint4*>(rp + c0)) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    const int t = tbeg + hw + 8 * r;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = m[r].x - 3 + j >= 0;
      const uint32_t xw[4] = {xv[r][j].x, xv[r][j].y, xv[r][j].z, xv[r][j].w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float f = fma_bf(wt[i][j], h16(xw[i >> 1], i & 1), acc[i]);
        acc[i] = ok ? f : acc[i];
      }
    }
    float s[8];
    bool fl = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = bfr(acc[i]);
      s[i] = bfr(silu_c_fast(y, fl));
    }
    if (__any_sync(0xffffffffu, fl)) {  // never taken for finite conv outputs > -87: the div.rn recipe
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float y = bfr(acc[i]);
        s[i] = bfr(div_rn(y, one_plus_exp_neg(y)));
      }
    }
    if (u < nqk) {
      float sum = mul_rn(s[1], s[1]);
      sum = fma_rn(s[0], s[0], sum);  // exact squares of bf16 values: same as Triton's fma.rn.f32.bf16
#pragma unroll
      for (int i = 2; i < 8; ++i) sum = fma_rn(s[i], s[i], sum);
      sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 8));
      sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 4));
      sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
      sum = add_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
      const float rstd = div_full(1.0f, sqrt_ftz(add_rn(eps, sum)));
      if (t >= Tf) continue;
      uint4 ov;
      ov.x = pk(mul_rn(rstd, s[0]), mul_rn(rstd, s[1]));
      ov.y = pk(mul_rn(rstd, s[2]), mul_rn(rstd, s[3]));
      ov.z = pk(mul_rn(rstd, s[4]), mul_rn(rstd, s[5]));
      ov.w = pk(mul_rn(rstd, s[6]), mul_rn(rstd, s[7]));
      const int nh = REP * KD / 128, hq = u % (KD / 128);
      __nv_bfloat16* dst = (u < KD / 128 ? q : k) + (long long)t * (nh * 128) + REP * hq * 128 + 8 * l16;
      *reinterpret_cast<uint4*>(dst) = ov;
      if (REP == 2) *reinterpret_cast<uint4*>(dst + 128) = ov;
    } else if (!SKIP_V) {
      if (t >= Tf) continue;
      uint4 ov;
      ov.x = pk(s[0], s[1]); ov.y = pk(s[2], s[3]); ov.z = pk(s[4], s[5]); ov.w = pk(s[6], s[7]);
      *reinterpret_cast<uint4*>(v + (long long)t * VD + (c0 - 2 * KD)) = ov;
    }
  }
}

template <int NR>
static void launch_conv(const __nv_bfloat16* a, long long SX, const __nv_bfloat16* cc, int C, const int* meta, const __nv_bfloat16* ww,
                        __nv_bfloat16* qq, __nv_bfloat16* kk, __nv_bfloat16* vv, int Tf, float eps, int KD, int VD, int rep, int skip_v,
                        cudaStream_t st, int u0 = 0, int nunits = -1) {
  const int units = nunits >= 0 ? nunits : 2 * KD / 128 + (skip_v ? 0 : VD / 128);
  dim3 grid((Tf + 8 * NR - 1) / (8 * NR), units);
  if (rep == 1 && !skip_v) conv_l2_kernel<1, false, NR><<<grid, 128, 0, st>>>(a, SX, cc, C, meta, ww, qq, kk, vv, Tf, eps, KD, VD, u0);
  else if (rep == 1) conv_l2_kernel<1, true, NR><<<grid, 128, 0, st>>>(a, SX, cc, C, meta, ww, qq, kk, vv, Tf, eps, KD, VD, u0);
  else if (!skip_v) conv_l2_kernel<2, false, NR><<<grid, 128, 0, st>>>(a, SX, cc, C, meta, ww, qq, kk, vv, Tf, eps, KD, VD, u0);
  else conv_l2_kernel<2, true, NR><<<grid, 128, 0, st>>>(a, SX, cc, C, meta, ww, qq, kk, vv, Tf, eps, KD, VD, u0);
}

extern "C" int reax_conv_l2(const void* qkv, long long SX, const void* cache, int C, const void* meta, const void* w, void* q,
                            void* k, void* v, int Tf, float eps, int KD, int VD, int rep, int skip_v, int RT, void* stream) {
  if (Tf <= 0) return 0;
  cudaStream_t st = (cudaStream_t)stream;
  const __nv_bfloat16* a = (const __nv_bfloat16*)qkv;
  const __nv_bfloat16* cc = (const __nv_bfloat16*)cache;
  const __nv_bfloat16* ww = (const __nv_bfloat16*)w;
  __nv_bfloat16 *qq = (__nv_bfloat16*)q, *kk = (__nv_bfloat16*)k, *vv = (__nv_bfloat16*)v;
  const int* mt = (const int*)meta;
  switch (RT) {  // RT = rows per CTA = 8 * rows per thread
    case 8: launch_conv<1>(a, SX, cc, C, mt, ww, qq, kk, vv, Tf, eps, KD, VD, rep, skip_v, st); break;
    case 16: launch_conv<2>(a, SX, cc, C, mt, ww, qq, kk, vv, Tf, eps, KD, VD, rep, skip_v, st); break;
    case 32: launch_conv<4>(a, SX, cc, C, mt, ww, qq, kk, vv, Tf, eps, KD, VD, rep, skip_v, st); break;
    default: return -1;
  }
  return (int)cudaGetLastError();
}

// value channels only (units 2*KD/128 .. +VD/128): v = SiLU(conv) for the v part, launched on a side stream
extern "C" int reax_conv_v(const void* qkv, long long SX, const void* cache, int C, const void* meta, const void* w, void* v, int Tf,
                           int KD, int VD, void* stream) {
  if (Tf <= 0) return 0;
  launch_conv<2>((const __nv_bfloat16*)qkv, SX, (const __nv_bfloat16*)cache, C, (const int*)meta, (const __nv_bfloat16*)w,
                 nullptr, nullptr, (__nv_bfloat16*)v, Tf, 1e-6f, KD, VD, 1, 0, (cudaStream_t)stream, 2 * KD / 128, VD / 128);
  return (int)cudaGetLastError();
}
