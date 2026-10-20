#define QSTREAM 1
#define FNW 4
#define FNST 2
#define FKC 64
#define FMINB 2
// fmha v11 (v10 with the attn_post epilogue staged through shared memory: coalesced 16-byte gate loads / output stores, fast sigmoid): exact replica of PyTorch mem-efficient attention forward (cutlassF bf16 32x128 gmem sm80, head dim 256).
// FA2-style (warp = 16 query rows x whole 128-key blocks x all 256 value dims; O in registers), cp.async K/V ring of
// 128-key x 64-dim chunks. Two mask modes:
//   QP = true : mask given as qpos (row r of batch b sees keys <= qpos[b, r / G]); values identical to the additive
//               0 / -inf bf16 bias built from the same qpos (accum*scale + 0 then *log2e for allowed, -inf otherwise)
//   QP = false: additive bf16 bias tensor (read per element), no skipping.
// Key work skipped exactly: 32-key groups beyond the warp's last visible key (probabilities exactly 0) and whole key
// blocks beyond the CTA's last visible key (reference combine emulated: O = rn(alpha*O) + 0 for the skipped last block).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#ifndef POST_ABL
#define POST_ABL 0
#endif
#ifndef QSTREAM
#define QSTREAM 0
#endif

#define HD 256
#define KB 128

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src, bool pred) {
#ifdef ABL_NOLOAD
  return;
#endif
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#ifdef ABL_NOMMA
  c[0] += __uint_as_float(a0 ^ b); return;
#endif
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((uint32_t)x << 16); }

__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float d; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float fma_rm_ftz(float a, float b, float c) { float d; asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
__device__ __forceinline__ float sat_ftz(float a) { float d; asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float add_rn(float a, float b) { float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float sub_rn(float a, float b) { float d; asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float mul_rn(float a, float b) { float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float div_rn(float a, float b) { float d; asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
__device__ __forceinline__ float ex2_ftz(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
// glue._sigmoid_f32 as compiled with enable_fp_fusion=False: div_rn(1, 1 + libdevice.exp(-x)), ftz expf path
__device__ __forceinline__ float sigmoid_post(float x) {
  const float a = sub_rn(0.f, x);
  float t = fma_ftz(a, __uint_as_float(0x3BBB989Du), 0.5f);
  t = sat_ftz(t);
  const float j = fma_rm_ftz(t, 252.0f, __uint_as_float(0x4B400001u));
  const float jj = add_rn(j, __uint_as_float(0xCB40007Fu));
  float f = fma_ftz(a, __uint_as_float(0x3FB8AA3Bu), -jj);
  f = fma_ftz(a, __uint_as_float(0x32A57060u), f);
  const float e = mul_rn(ex2_ftz(f), __uint_as_float(__float_as_uint(j) << 23));
  return div_rn(1.0f, add_rn(e, 1.0f));
}
__device__ __forceinline__ float rcp_approx(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
// 1/d correctly rounded for d in [1, 2^126) (Markstein: refined reciprocal, exact residual, final fma); d = 1 + e
__device__ __forceinline__ float rcp_rn_fast(float d) {
  const float r0 = rcp_approx(d);
  const float r1 = fma_rn(r0, fma_rn(-d, r0, 1.0f), r0);
  return fma_rn(r1, fma_rn(-d, r1, 1.0f), r1);
}
// sigmoid_post with the final division as rcp_rn_fast (valid while 1 + e < 2^126, i.e. gate > ~-87.3)
__device__ __forceinline__ float sigmoid_post_d(float x) {
  const float a = sub_rn(0.f, x);
  float t = fma_ftz(a, __uint_as_float(0x3BBB989Du), 0.5f);
  t = sat_ftz(t);
  const float j = fma_rm_ftz(t, 252.0f, __uint_as_float(0x4B400001u));
  const float jj = add_rn(j, __uint_as_float(0xCB40007Fu));
  float f = fma_ftz(a, __uint_as_float(0x3FB8AA3Bu), -jj);
  f = fma_ftz(a, __uint_as_float(0x32A57060u), f);
  const float e = mul_rn(ex2_ftz(f), __uint_as_float(__float_as_uint(j) << 23));
  return add_rn(e, 1.0f);
}
__device__ __forceinline__ float ex2_approx(float a) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
// sigmoid fast path s' = rcp(1 + 2^(-x log2e)) (MUFU approximations). Exhaustively checked over all 65536 bf16 gates
// (rv2/cuda/sig_all_test.py, sig_nomid.py): bf16(s') == bf16(sigmoid_post(x)) bitwise for every gate except
// x in {-87.5, -88, -88.5}, where the exact result is a bf16 subnormal and rcp.approx.ftz flushes; those (s' < 2^-125,
// gate <= -87) are flagged and recomputed with the exact recipe.
__device__ __forceinline__ float sig_fast(float x, bool& unsafe) {
  float s;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(__fadd_rn(ex2_approx(__fmul_rn(x, -1.4426950408889634f)), 1.0f)));
  const uint32_t u = __float_as_uint(s);
  unsafe |= u < 0x01000000u;
  return s;
}
__device__ __forceinline__ uint32_t mul_bf2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
// physical 16-byte unit of logical unit u in a chunk row r (UR units per row): conflict-free ldmatrix on 8 rows
template <int UR>
__device__ __forceinline__ int swz(int u, int r) { return UR == 8 ? (u ^ (r & 7)) : (u ^ ((r >> 1) & 3)); }

// S tiles 4g..4g+3 for k-steps of one 64-dim K chunk (k-step pairs j, j+1; reference order: ascending k per element)
template <int NG, int UR>
__device__ __forceinline__ void mm0_chunk(float (&acc)[16][4], uint32_t qrow_base, int qsw, uint32_t st, int kc, int lane) {
  const int mi = lane >> 3;
#pragma unroll
  for (int j = 0; j < UR; j += 2) {
    uint32_t a[4];
    {
#if QSTREAM
      const int u = j + (mi >> 1);
      ldsm4(a, qrow_base + st + (uint32_t)(swz<UR>(u, qsw) * 16));
#else
      const int u = kc * UR + j + (mi >> 1);
      ldsm4(a, qrow_base + (uint32_t)((u ^ qsw) * 16));
#endif
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      uint32_t bk[2][4];
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int key = 8 * (4 * g + 2 * p + (mi >> 1)) + (lane & 7), u = j + (mi & 1);
        ldsm4(bk[p], st + (uint32_t)(key * UR + swz<UR>(u, key)) * 16);
      }
      mma8(acc[4 * g + 0], a[0], a[1], bk[0][0]);
      mma8(acc[4 * g + 1], a[0], a[1], bk[0][2]);
      mma8(acc[4 * g + 2], a[0], a[1], bk[1][0]);
      mma8(acc[4 * g + 3], a[0], a[1], bk[1][2]);
      mma8(acc[4 * g + 0], a[2], a[3], bk[0][1]);
      mma8(acc[4 * g + 1], a[2], a[3], bk[0][3]);
      mma8(acc[4 * g + 2], a[2], a[3], bk[1][1]);
      mma8(acc[4 * g + 3], a[2], a[3], bk[1][3]);
    }
  }
}

// O_blk for 4 dim tiles (4dg..4dg+3 of the chunk) over key groups 0..NG-1, each tile's chain in ascending key order
template <int NG, int UR>
__device__ __forceinline__ void mm1_group(float (&ob)[4][4], const uint32_t (&pa)[16][2], uint32_t st, int dg, int lane) {
#pragma unroll
  for (int jg = 0; jg < NG; ++jg) {
    uint32_t bv[4][4];
    const int key = 32 * jg + lane;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int u = 4 * dg + t;
      ldsm4t(bv[t], st + (uint32_t)(key * UR + swz<UR>(u, key)) * 16);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int t = 0; t < 4; ++t) mma8(ob[t], pa[4 * jg + s][0], pa[4 * jg + s][1], bv[t][s]);
  }
}

template <int NW, int NST, int KC, int MINB, bool QP, bool POST>
__global__ void __launch_bounds__(NW * 32, MINB)
fmha3_kernel(const __nv_bfloat16* __restrict__ Qp, const __nv_bfloat16* __restrict__ Kp, const __nv_bfloat16* __restrict__ Vp,
             const unsigned short* __restrict__ Bp, const int* __restrict__ qpos, int G, long long qpBs,
             __nv_bfloat16* __restrict__ Op, float scale, int M, int N,
             long long qsB, long long qsH, long long qsM, long long ksB, long long ksH, long long ksM,
             long long vsB, long long vsH, long long vsM, long long bsB, long long bsH, long long bsM,
             long long osB, long long osH, long long osM, const int* __restrict__ inv, int Lq,
             const __nv_bfloat16* __restrict__ qg, long long qg_rs, __nv_bfloat16* __restrict__ outp, long long out_rs) {
  extern __shared__ __align__(128) unsigned char smraw[];
  __shared__ int cta_qmax[NW];
  constexpr int NT = NW * 32, UR = KC / 8, NCH = HD / KC;
#if QSTREAM
  constexpr uint32_t KVB = KB * UR * 16, STAGE = KVB + NW * 16 * UR * 16;  // [K/V chunk][Q chunk]
  const uint32_t qbase = sa(smraw), kvbase = qbase;
#else
  constexpr uint32_t STAGE = KB * UR * 16;
  const uint32_t qbase = sa(smraw), kvbase = qbase + NW * 16 * 32 * 16;
#endif
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  // grid (H, B, tiles): the tile index is the slowest-dispatched dimension, latest (most keys) tiles first
  const int q0 = (gridDim.z - 1 - blockIdx.z) * (NW * 16), h = blockIdx.x, b = blockIdx.y;
  const __nv_bfloat16* Q = Qp + b * qsB + h * qsH;
  const __nv_bfloat16* K = Kp + b * ksB + h * ksH;
  const __nv_bfloat16* V = Vp + b * vsB + h * vsH;
  const int nkb = (N + KB - 1) / KB;

  const int wr0 = q0 + warp * 16, rowA = wr0 + gq, rowB = rowA + 8;
  const bool wlive = wr0 < M, okA = rowA < M, okB = rowB < M;
  int qA = -1, qB = -1;
  const unsigned short* bA = nullptr;
  const unsigned short* bB = nullptr;
  if (QP) {
    if (okA) qA = qpos[b * qpBs + rowA / G];
    if (okB) qB = qpos[b * qpBs + rowB / G];
  } else {
    const unsigned short* Bi = Bp + b * bsB + h * bsH;
    bA = Bi + (long long)(okA ? rowA : 0) * bsM;
    bB = Bi + (long long)(okB ? rowB : 0) * bsM;
  }
  const int wq = QP ? __reduce_max_sync(0xffffffffu, max(qA, qB)) : 0;
  int nkb_c = nkb, cq = -1;
  if (QP) {
    if (lane == 0) cta_qmax[warp] = wq;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < NW; ++w) cq = max(cq, cta_qmax[w]);
    nkb_c = cq <= 0 ? 0 : min(nkb, cq / KB + 1);
  }
  const int total = 2 * NCH * nkb_c;

#if !QSTREAM
  for (int i = tid; i < NW * 16 * 32; i += NT) {
    const int r = i >> 5, u = i & 31;
    const bool ok = q0 + r < M;
    cpa16(qbase + (uint32_t)(r * 32 + (u ^ (r & 7))) * 16, Q + (long long)(ok ? q0 + r : 0) * qsM + u * 8, ok);
  }
#endif
  auto load_chunk = [&](int c) {
    const int kb = c / (2 * NCH), idx = c % (2 * NCH);
    const bool isv = idx >= NCH;
    const int d0 = (isv ? idx - NCH : idx) * KC;
    const __nv_bfloat16* base = isv ? V : K;
    const long long sM = isv ? vsM : ksM;
    const uint32_t dst = kvbase + (uint32_t)(c % NST) * STAGE;
#pragma unroll
    for (int i = tid; i < KB * UR; i += NT) {
      const int r = i / UR, u = i % UR, key = kb * KB + r;
      const bool ok = key < N;
      cpa16(dst + (uint32_t)(r * UR + swz<UR>(u, r)) * 16, base + (long long)(ok ? key : 0) * sM + d0 + u * 8, ok);
    }
#if QSTREAM
    if (!isv) {
#pragma unroll
      for (int i = tid; i < NW * 16 * UR; i += NT) {
        const int r = i / UR, u = i % UR;
        const bool ok = q0 + r < M;
        cpa16(dst + KVB + (uint32_t)(r * UR + swz<UR>(u, r)) * 16, Q + (long long)(ok ? q0 + r : 0) * qsM + d0 + u * 8, ok);
      }
    }
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < total) load_chunk(s);
    cp_commit();
  }
  auto step = [&](int c) {
    cp_wait<NST - 2>();
    __syncthreads();
    if (c + NST - 1 < total) load_chunk(c + NST - 1);
    cp_commit();
    return kvbase + (uint32_t)(c % NST) * STAGE;
  };
  const int qr = warp * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
#if QSTREAM
  const uint32_t qrow_base = KVB + (uint32_t)(qr * UR * 16);  // + stage address at use
  const int qsw = qr;                                          // row index for swz
#else
  const uint32_t qrow_base = qbase + (uint32_t)(qr * 32 * 16);
  const int qsw = qr & 7;
#endif
  const float kLog2e = 1.4426950408889634074f;

  float o[32][4];
#pragma unroll
  for (int n = 0; n < 32; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[n][e] = 0.f;
  float mA = -INFINITY, mB = -INFINITY, sA = 0.f, sB = 0.f;
  if (QP && cq == 0) {
    // every row here sees only key 0: P = exp2(0) = 1, s = 1, O_blk = 1*V[0] exactly, O = 1*O_blk -> V[0]; the
    // remaining (masked) blocks are applied by the skipped-block emulation below (alpha = 1, rn(O) + 0)
    const unsigned int* v0 = reinterpret_cast<const unsigned int*>(V);
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      const unsigned int x = v0[4 * n + tq];
      o[n][0] = o[n][2] = __uint_as_float(x << 16);
      o[n][1] = o[n][3] = __uint_as_float(x & 0xffff0000u);
    }
    sA = sB = 1.f;
  }

  for (int kb = 0; kb < nkb_c; ++kb) {
    const int ks = kb * KB, nk = min(KB, N - ks);
    const bool isFirst = kb == 0, isLast = kb == nkb - 1;
    int vis = nk;
    if (QP) vis = max(0, min(nk, wq + 1 - ks));
    const int ng = wlive ? (vis + 31) / 32 : 0;
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[n][e] = 0.f;
    for (int kc = 0; kc < NCH; ++kc) {
      const uint32_t st = step(kb * 2 * NCH + kc);
      switch (ng) {
        case 4: mm0_chunk<4, UR>(acc, qrow_base, qsw, st, kc, lane); break;
        case 3: mm0_chunk<3, UR>(acc, qrow_base, qsw, st, kc, lane); break;
        case 2: mm0_chunk<2, UR>(acc, qrow_base, qsw, st, kc, lane); break;
        case 1: mm0_chunk<1, UR>(acc, qrow_base, qsw, st, kc, lane); break;
        default: break;
      }
    }
    float eA = 1.f, eB = 1.f, alA = 1.f, alB = 1.f, beA = 1.f, beB = 1.f;
    uint32_t pa[16][2];
#ifdef ABL_NOSOFT
    for (int n = 0; n < 16; ++n) { pa[n][0] = __float_as_uint(acc[n][0]); pa[n][1] = __float_as_uint(acc[n][2]); }
    if (false) {
#else
    if (wlive) {
#endif
      float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        if (n >= 4 * ng) continue;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int col = 8 * n + 2 * tq + c;
          float xa = -INFINITY, xb = -INFINITY;
          if (col < nk) {
            if (QP) {
              if (ks + col <= qA) xa = __fmul_rn(__fadd_rn(__fmul_rn(acc[n][c], scale), 0.f), kLog2e);
              if (ks + col <= qB) xb = __fmul_rn(__fadd_rn(__fmul_rn(acc[n][2 + c], scale), 0.f), kLog2e);
            } else {
              xa = __fmul_rn(__fadd_rn(__fmul_rn(acc[n][c], scale), okA ? bf(bA[ks + col]) : 0.f), kLog2e);
              xb = __fmul_rn(__fadd_rn(__fmul_rn(acc[n][2 + c], scale), okB ? bf(bB[ks + col]) : 0.f), kLog2e);
            }
            mxA = fmaxf(mxA, xa);
            mxB = fmaxf(mxB, xb);
          }
          acc[n][c] = xa;
          acc[n][2 + c] = xb;
        }
      }
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 1));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 1));
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
      const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
      const bool chA = mA < nA, chB = mB < nB;
      if (chA) {
        eA = exp2f(mA - nA);
        sA = __fmul_rn(sA, eA);
      }
      if (chB) {
        eB = exp2f(mB - nB);
        sB = __fmul_rn(sB, eB);
      }
      const float uA = (!chA && nA == -INFINITY) ? 0.f : nA, uB = (!chB && nB == -INFINITY) ? 0.f : nB;
      mA = nA;
      mB = nB;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        if (w >= ng) continue;
        float ta = 0.f, tb = 0.f;
#pragma unroll
        for (int n = 4 * w; n < 4 * w + 4; ++n)
#pragma unroll
          for (int c = 0; c < 2; ++c) {
#ifdef ABL_NOEXP
            const float pa_ = acc[n][c] - uA, pb_ = acc[n][2 + c] - uB;
#else
            const float pa_ = exp2f(acc[n][c] - uA), pb_ = exp2f(acc[n][2 + c] - uB);
#endif
            acc[n][c] = pa_;
            acc[n][2 + c] = pb_;
            ta = __fadd_rn(ta, pa_);
            tb = __fadd_rn(tb, pb_);
          }
        ta = __fadd_rn(ta, __shfl_xor_sync(0xffffffffu, ta, 1));
        tb = __fadd_rn(tb, __shfl_xor_sync(0xffffffffu, tb, 1));
        ta = __fadd_rn(ta, __shfl_xor_sync(0xffffffffu, ta, 2));
        tb = __fadd_rn(tb, __shfl_xor_sync(0xffffffffu, tb, 2));
        sA = __fadd_rn(sA, ta);
        sB = __fadd_rn(sB, tb);
      }
      if (isLast) {
        alA = __fdiv_rn(1.f, sA == 0.f ? 1.f : sA);
        alB = __fdiv_rn(1.f, sB == 0.f ? 1.f : sB);
      }
      beA = __fmul_rn(alA, eA);
      beB = __fmul_rn(alB, eB);
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        pa[n][0] = pk(acc[n][0], acc[n][1]);
        pa[n][1] = pk(acc[n][2], acc[n][3]);
      }
    }
#pragma unroll
    for (int vc = 0; vc < NCH; ++vc) {
      const uint32_t st = step(kb * 2 * NCH + NCH + vc);
      if (wlive) {
#pragma unroll
        for (int dg = 0; dg < UR / 4; ++dg) {
          float ob[4][4];
#pragma unroll
          for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) ob[t][e] = 0.f;
          switch (ng) {
            case 4: mm1_group<4, UR>(ob, pa, st, dg, lane); break;
            case 3: mm1_group<3, UR>(ob, pa, st, dg, lane); break;
            case 2: mm1_group<2, UR>(ob, pa, st, dg, lane); break;
            case 1: mm1_group<1, UR>(ob, pa, st, dg, lane); break;
            default: break;
          }
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            float* O = o[vc * UR + 4 * dg + t];
#ifdef ABL_NOCOMB
            for (int e = 0; e < 4; ++e) O[e] += ob[t][e];
            if (false)
#endif
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float ee = e < 2 ? eA : eB, al = e < 2 ? alA : alB, be = e < 2 ? beA : beB;
              if (isFirst) O[e] = isLast ? __fmul_rn(al, ob[t][e]) : ob[t][e];
              else if (!isLast) O[e] = __fmaf_rn(ee, O[e], ob[t][e]);
              else O[e] = __fmaf_rn(ob[t][e], al, __fmul_rn(be, O[e]));
            }
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (POST) __syncthreads();  // every warp is done with the K/V stages: the ring is reused to stage O
  if (!wlive) return;
  if (nkb_c < nkb) {  // skipped key blocks (fully masked for every row here) incl. the last one: O = rn(alpha*O) + 0
    const float a2 = __fdiv_rn(1.f, sA == 0.f ? 1.f : sA), b2 = __fdiv_rn(1.f, sB == 0.f ? 1.f : sB);
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      o[n][0] = __fadd_rn(__fmul_rn(a2, o[n][0]), 0.f);
      o[n][1] = __fadd_rn(__fmul_rn(a2, o[n][1]), 0.f);
      o[n][2] = __fadd_rn(__fmul_rn(b2, o[n][2]), 0.f);
      o[n][3] = __fadd_rn(__fmul_rn(b2, o[n][3]), 0.f);
    }
  }
  if (POST) {
    // attn_post fused: fold row r = m*G + g of (b, kv head h) -> query head h*G + g of token row inv[b*Lq + m].
    // O (bf16) of the warp's 16 rows -> smem (row i at 512 B, 16-byte unit u stored at u ^ (i & 7)); then one row per
    // step, lane l owns dims 8l..8l+7: coalesced 16-byte gate load / output store.
    const uint32_t ob = sa(smraw) + (uint32_t)warp * (16 * HD * 2);
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      const uint32_t lo = pk(o[n][0], o[n][1]), hi = pk(o[n][2], o[n][3]);
      asm volatile("st.shared.b32 [%0], %1;" ::"r"(ob + (uint32_t)(gq * 512 + ((n ^ gq) * 16) + tq * 4)), "r"(lo) : "memory");
      asm volatile("st.shared.b32 [%0], %1;" ::"r"(ob + (uint32_t)((gq + 8) * 512 + ((n ^ gq) * 16) + tq * 4)), "r"(hi) : "memory");
    }
    int Rl = -1;
    if (lane < 16 && wr0 + lane < M) Rl = inv[(long long)b * Lq + (wr0 + lane) / G];
    __syncwarp();
    uint4 gv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int R = __shfl_sync(0xffffffffu, Rl, i);
      gv[i] = make_uint4(0u, 0u, 0u, 0u);
      if (R >= 0)
        gv[i] = *reinterpret_cast<const uint4*>(qg + (long long)R * qg_rs + (long long)(h * G + (wr0 + i) % G) * (2 * HD) + HD + 8 * lane);
    }
    // fast sigmoid for every row (no per-row branch: keeps the 16 rows' MUFU chains interleaved); rows whose bf16
    // rounding the fast path cannot decide are marked in umask and redone with the exact recipe after the loop
    uint32_t umask = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int R = __shfl_sync(0xffffffffu, Rl, i);
      if (R < 0) continue;
      uint4 ov;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(ov.x), "=r"(ov.y), "=r"(ov.z), "=r"(ov.w)
                   : "r"(ob + (uint32_t)(i * 512 + ((lane ^ (i & 7)) * 16))) : "memory");
      const uint32_t g4[4] = {gv[i].x, gv[i].y, gv[i].z, gv[i].w};
      const uint32_t o4[4] = {ov.x, ov.y, ov.z, ov.w};
      uint32_t r4[4];
      bool uns = false;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#if POST_ABL & 1
        const uint32_t sg = g4[e];
#else
        const float lo = sig_fast(bf((unsigned short)(g4[e] & 0xffffu)), uns);
        const float hi = sig_fast(bf((unsigned short)(g4[e] >> 16)), uns);
        const uint32_t sg = pk(lo, hi);
#endif
        r4[e] = mul_bf2(o4[e], sg);
      }
      umask |= uns ? (1u << i) : 0u;
      *reinterpret_cast<uint4*>(outp + (long long)R * out_rs + (long long)(h * G + (wr0 + i) % G) * HD + 8 * lane) =
          make_uint4(r4[0], r4[1], r4[2], r4[3]);
    }
#if !(POST_ABL & 1)
    if (__any_sync(0xffffffffu, umask != 0u)) {  // rare: exact recipe for the marked (lane, row) pieces, rewritten
#pragma unroll 1
      for (int i = 0; i < 16; ++i) {
        const int R = __shfl_sync(0xffffffffu, Rl, i);
        if (R < 0 || !((umask >> i) & 1u)) continue;
        const long long hq = h * G + (wr0 + i) % G;
        const uint4 gw = *reinterpret_cast<const uint4*>(qg + (long long)R * qg_rs + hq * (2 * HD) + HD + 8 * lane);
        uint4 ov;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(ov.x), "=r"(ov.y), "=r"(ov.z), "=r"(ov.w)
                     : "r"(ob + (uint32_t)(i * 512 + ((lane ^ (i & 7)) * 16))) : "memory");
        const uint32_t g4[4] = {gw.x, gw.y, gw.z, gw.w};
        const uint32_t o4[4] = {ov.x, ov.y, ov.z, ov.w};
        uint32_t r4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          r4[e] = mul_bf2(o4[e], pk(sigmoid_post(bf((unsigned short)(g4[e] & 0xffffu))), sigmoid_post(bf((unsigned short)(g4[e] >> 16)))));
        *reinterpret_cast<uint4*>(outp + (long long)R * out_rs + hq * HD + 8 * lane) = make_uint4(r4[0], r4[1], r4[2], r4[3]);
      }
    }
#endif
  } else {
    __nv_bfloat16* Ob = Op + b * osB + h * osH;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      const int col = 8 * n + 2 * tq;
      if (okA) *reinterpret_cast<uint32_t*>(Ob + (long long)rowA * osM + col) = pk(o[n][0], o[n][1]);
      if (okB) *reinterpret_cast<uint32_t*>(Ob + (long long)rowB * osM + col) = pk(o[n][2], o[n][3]);
    }
  }
}

#ifndef FNW
#define FNW 8
#endif
#ifndef FNST
#define FNST 2
#endif
#ifndef FKC
#define FKC 64
#endif
#ifndef FMINB
#define FMINB 1
#endif
template <bool QP, bool POST>
static int launch2(const void* q, const void* k, const void* v, const void* bias, const int* qpos, int G, long long qpBs, void* o,
                   float scale, int B, int H, int M, int N, long long qsB, long long qsH, long long qsM, long long ksB,
                   long long ksH, long long ksM, long long vsB, long long vsH, long long vsM, long long bsB, long long bsH,
                   long long bsM, long long osB, long long osH, long long osM, cudaStream_t st, const int* inv = nullptr,
                   int Lq = 0, const void* qg = nullptr, long long qg_rs = 0, void* outp = nullptr, long long out_rs = 0) {
  constexpr int SMEM = FNST * (KB * (FKC / 8) * 16 + FNW * 16 * (FKC / 8) * 16);
  static_assert(!POST || SMEM >= FNW * 16 * HD * 2, "POST stages the warp's O tile in the K/V ring");
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(fmha3_kernel<FNW, FNST, FKC, FMINB, QP, POST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    init = true;
  }
  dim3 grid(H, B, (M + FNW * 16 - 1) / (FNW * 16));
  fmha3_kernel<FNW, FNST, FKC, FMINB, QP, POST><<<grid, FNW * 32, SMEM, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
      (const __nv_bfloat16*)v, (const unsigned short*)bias, qpos, G, qpBs, (__nv_bfloat16*)o, scale, M, N, qsB, qsH, qsM, ksB,
      ksH, ksM, vsB, vsH, vsM, bsB, bsH, bsM, osB, osH, osM, inv, Lq, (const __nv_bfloat16*)qg, qg_rs, (__nv_bfloat16*)outp, out_rs);
  return (int)cudaGetLastError();
}
// bias mode (same signature as fmha0/fmha1)
extern "C" int reax_fmha_b(const void* q, const void* k, const void* v, const void* bias, void* o, float scale, int B, int H, int M, int N,
                     long long qsB, long long qsH, long long qsM, long long ksB, long long ksH, long long ksM,
                     long long vsB, long long vsH, long long vsM, long long bsB, long long bsH, long long bsM,
                     long long osB, long long osH, long long osM, void* stream) {
  if (!bias) return -1;
  return launch2<false, false>(q, k, v, bias, nullptr, 1, 0, o, scale, B, H, M, N, qsB, qsH, qsM, ksB, ksH, ksM, vsB, vsH, vsM, bsB,
                        bsH, bsM, osB, osH, osM, (cudaStream_t)stream);
}
// qpos mode: row r of batch b sees keys 0..qpos[b * qpBs + r / G]
extern "C" int reax_fmha_q(const void* q, const void* k, const void* v, const int* qpos, int G, long long qpBs, void* o, float scale,
                      int B, int H, int M, int N, long long qsB, long long qsH, long long qsM, long long ksB, long long ksH,
                      long long ksM, long long vsB, long long vsH, long long vsM, long long osB, long long osH, long long osM,
                      void* stream) {
  return launch2<true, false>(q, k, v, nullptr, qpos, G, qpBs, o, scale, B, H, M, N, qsB, qsH, qsM, ksB, ksH, ksM, vsB, vsH, vsM, 0, 0,
                       0, osB, osH, osM, (cudaStream_t)stream);
}

// qpos mode with the attn_post epilogue fused: out[inv[b*Lq + r/G]][(h*G + r%G)*D + d] = bf16(o * bf16(sigmoid(gate)))
extern "C" int reax_fmha_qp(const void* q, const void* k, const void* v, const int* qpos, int G, long long qpBs, float scale, int B,
                       int H, int M, int N, long long qsB, long long qsH, long long qsM, long long ksB, long long ksH,
                       long long ksM, long long vsB, long long vsH, long long vsM, const int* inv, int Lq, const void* qg,
                       long long qg_rs, void* outp, long long out_rs, void* stream) {
  return launch2<true, true>(q, k, v, nullptr, qpos, G, qpBs, nullptr, scale, B, H, M, N, qsB, qsH, qsM, ksB, ksH, ksM, vsB,
                             vsH, vsM, 0, 0, 0, 0, 0, 0, (cudaStream_t)stream, inv, Lq, qg, qg_rs, outp, out_rs);
}
