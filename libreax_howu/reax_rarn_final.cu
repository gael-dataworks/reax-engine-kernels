// Exact CUDA port of engine.norms._reduce_add_rmsnorm (bf16 partials, D = 2560), 2 warps per row:
// thread t of a row owns lane-slot l = t>>1 and the two accumulators v = 2*(t&1), 2*(t&1)+1 (cols 128k + 4l + v).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define DCH 20
#ifndef RPF
#define RPF 10
#endif

__device__ __forceinline__ float blo(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float bhi(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int S, bool SERIAL>
__global__ void __launch_bounds__(256)
rarn2_kernel(const __nv_bfloat16* __restrict__ part, long long pstride, const __nv_bfloat16* __restrict__ res,
             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ out,
             int n, float eps, float factor) {
  __shared__ float red[4][17];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rl = warp >> 1, t = tid & 63;          // row within CTA, thread within row
  const int row = blockIdx.x * 4 + rl;
  const bool live = row < n;
  const int sl = t >> 1, hf2 = t & 1;              // lane-slot, half
  const long long ro = (long long)(live ? row : 0) * (128 * DCH) + 4 * sl + 2 * hf2;
  float a0 = 0.f, a1 = 0.f;
  uint32_t hv[DCH];
#pragma unroll
  for (int k0 = 0; k0 < DCH; k0 += RPF) {
    uint32_t p[RPF][S], rr[RPF];
#pragma unroll
    for (int c = 0; c < RPF; ++c) {
#pragma unroll
      for (int s = 0; s < S; ++s) p[c][s] = live ? *reinterpret_cast<const unsigned int*>(part + s * pstride + ro + 128 * (k0 + c)) : 0u;
      rr[c] = live ? *reinterpret_cast<const uint32_t*>(res + ro + 128 * (k0 + c)) : 0u;
    }
#pragma unroll
    for (int c = 0; c < RPF; ++c) {
      float hh[2];
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        float d = v ? bhi(p[c][0]) : blo(p[c][0]);
        if (SERIAL) d = bfr(d);
#pragma unroll
        for (int s = 1; s < S; ++s) {
          d = __fadd_rn(d, v ? bhi(p[c][s]) : blo(p[c][s]));
          if (SERIAL) d = bfr(d);
        }
        d = bfr(d);
        hh[v] = bfr(__fadd_rn(v ? bhi(rr[c]) : blo(rr[c]), d));
      }
      a0 = __fadd_rn(a0, __fmul_rn(hh[0], hh[0]));
      a1 = __fadd_rn(a1, __fmul_rn(hh[1], hh[1]));
      hv[k0 + c] = pk(hh[0], hh[1]);
      if (live) *reinterpret_cast<uint32_t*>(h_out + ro + 128 * (k0 + c)) = hv[k0 + c];
    }
  }
  // per lane-slot: ((acc0 + acc1) + acc2) + acc3 ; acc0/1 in the even thread, acc2/3 in the odd thread
  const float e01 = __fadd_rn(a0, a1);                                  // valid in even threads
  const float from_even = __shfl_sync(0xffffffffu, e01, lane & ~1);
  float s = __fadd_rn(__fadd_rn(from_even, a0), a1);                    // valid in odd threads (a0, a1 = acc2, acc3)
  // half-split tree over 32 slots: slot m + slot m+16 (slots 16..31 live in the second warp of the row)
  if ((warp & 1) && (lane & 1)) red[rl][lane >> 1] = s;
  __syncthreads();
  float tot = 0.f;
  if (!(warp & 1)) {
    float x = (lane & 1) ? __fadd_rn(s, red[rl][lane >> 1]) : 0.f;   // slots m (m<16) at odd lanes 2m+1
    x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, 16));          // m + m+8
    x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, 8));           // m + m+4
    x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, 4));           // m + m+2
    x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, 2));           // m + m+1
    if (lane == 1) red[rl][16] = x;
  }
  __syncthreads();
  tot = red[rl][16];
  if (!live) return;
  const float r = rsqrtf(__fadd_rn(__fmul_rn(tot, factor), eps));
#pragma unroll
  for (int k = 0; k < DCH; ++k) {
    const uint32_t w2 = *reinterpret_cast<const uint32_t*>(w + 128 * k + 4 * sl + 2 * hf2);
    const float o0 = __fmul_rn(__fmul_rn(blo(hv[k]), r), __fadd_rn(1.f, blo(w2)));
    const float o1 = __fmul_rn(__fmul_rn(bhi(hv[k]), r), __fadd_rn(1.f, bhi(w2)));
    *reinterpret_cast<uint32_t*>(out + ro + 128 * k) = pk(o0, o1);
  }
}

template <int S, bool SER>
static int go2(const void* part, long long ps, const void* res, const void* w, void* h, void* o, int n, float eps, float f, cudaStream_t st) {
  rarn2_kernel<S, SER><<<(n + 3) / 4, 256, 0, st>>>((const __nv_bfloat16*)part, ps, (const __nv_bfloat16*)res, (const __nv_bfloat16*)w,
                                                   (__nv_bfloat16*)h, (__nv_bfloat16*)o, n, eps, f);
  return (int)cudaGetLastError();
}
extern "C" int reax_rarn(const void* part, int S, int serial, long long pstride, const void* res, const void* w, void* h_out,
                         void* out, int n, int D, float eps, float factor, void* stream) {
  if (D != 128 * DCH || n <= 0) return n <= 0 ? 0 : -1;
  cudaStream_t st = (cudaStream_t)stream;
#define CASE(SS) case SS: return serial ? go2<SS, true>(part, pstride, res, w, h_out, out, n, eps, factor, st) \
                                         : go2<SS, false>(part, pstride, res, w, h_out, out, n, eps, factor, st);
  switch (S) { CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) default: return -2; }
#undef CASE
}
