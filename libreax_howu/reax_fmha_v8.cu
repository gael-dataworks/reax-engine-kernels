#define QSTREAM 1
#define FNW 4
#define FNST 2
#define FKC 64
#define FMINB 2
// fmha v8 (v5 + CTAs whose rows all see only key 0 write V[0] directly): exact replica of PyTorch mem-efficient attention forward (cutlassF bf16 32x128 gmem sm80, head dim 256).
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

template <int NW, int NST, int KC, int MINB, bool QP>
__global__ void __launch_bounds__(NW * 32, MINB)
fmha3_kernel(const __nv_bfloat16* __restrict__ Qp, const __nv_bfloat16* __restrict__ Kp, const __nv_bfloat16* __restrict__ Vp,
             const unsigned short* __restrict__ Bp, const int* __restrict__ qpos, int G, long long qpBs,
             __nv_bfloat16* __restrict__ Op, float scale, int M, int N,
             long long qsB, long long qsH, long long qsM, long long ksB, long long ksH, long long ksM,
             long long vsB, long long vsH, long long vsM, long long bsB, long long bsH, long long bsM,
             long long osB, long long osH, long long osM) {
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
  __nv_bfloat16* Ob = Op + b * osB + h * osH;
#pragma unroll
  for (int n = 0; n < 32; ++n) {
    const int col = 8 * n + 2 * tq;
    if (okA) *reinterpret_cast<uint32_t*>(Ob + (long long)rowA * osM + col) = pk(o[n][0], o[n][1]);
    if (okB) *reinterpret_cast<uint32_t*>(Ob + (long long)rowB * osM + col) = pk(o[n][2], o[n][3]);
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
template <bool QP>
static int launch2(const void* q, const void* k, const void* v, const void* bias, const int* qpos, int G, long long qpBs, void* o,
                   float scale, int B, int H, int M, int N, long long qsB, long long qsH, long long qsM, long long ksB,
                   long long ksH, long long ksM, long long vsB, long long vsH, long long vsM, long long bsB, long long bsH,
                   long long bsM, long long osB, long long osH, long long osM, cudaStream_t st) {
#if QSTREAM
  constexpr int SMEM = FNST * (KB * (FKC / 8) * 16 + FNW * 16 * (FKC / 8) * 16);
#else
  constexpr int SMEM = FNW * 16 * 32 * 16 + FNST * KB * (FKC / 8) * 16;
#endif
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(fmha3_kernel<FNW, FNST, FKC, FMINB, QP>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    init = true;
  }
  dim3 grid(H, B, (M + FNW * 16 - 1) / (FNW * 16));
  fmha3_kernel<FNW, FNST, FKC, FMINB, QP><<<grid, FNW * 32, SMEM, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
      (const __nv_bfloat16*)v, (const unsigned short*)bias, qpos, G, qpBs, (__nv_bfloat16*)o, scale, M, N, qsB, qsH, qsM, ksB,
      ksH, ksM, vsB, vsH, vsM, bsB, bsH, bsM, osB, osH, osM);
  return (int)cudaGetLastError();
}
// bias mode (same signature as fmha0/fmha1)
extern "C" int reax_fmha_b(const void* q, const void* k, const void* v, const void* bias, void* o, float scale, int B, int H, int M, int N,
                     long long qsB, long long qsH, long long qsM, long long ksB, long long ksH, long long ksM,
                     long long vsB, long long vsH, long long vsM, long long bsB, long long bsH, long long bsM,
                     long long osB, long long osH, long long osM, void* stream) {
  if (!bias) return -1;
  return launch2<false>(q, k, v, bias, nullptr, 1, 0, o, scale, B, H, M, N, qsB, qsH, qsM, ksB, ksH, ksM, vsB, vsH, vsM, bsB,
                        bsH, bsM, osB, osH, osM, (cudaStream_t)stream);
}
// qpos mode: row r of batch b sees keys 0..qpos[b * qpBs + r / G]
extern "C" int reax_fmha_q(const void* q, const void* k, const void* v, const int* qpos, int G, long long qpBs, void* o, float scale,
                      int B, int H, int M, int N, long long qsB, long long qsH, long long qsM, long long ksB, long long ksH,
                      long long ksM, long long vsB, long long vsH, long long vsM, long long osB, long long osH, long long osM,
                      void* stream) {
  return launch2<true>(q, k, v, nullptr, qpos, G, qpBs, o, scale, B, H, M, N, qsB, qsH, qsM, ksB, ksH, ksM, vsB, vsH, vsM, 0, 0,
                       0, osB, osH, osM, (cudaStream_t)stream);
}
