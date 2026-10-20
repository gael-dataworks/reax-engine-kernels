// kkt v3: exact CUDA port of FLA chunk_gated_delta_rule_fwd_kkt_solve_kernel (BT=64, BC=16, K=128, USE_G, varlen, HV=2H).
// One CTA per (chunk, key head), 8 warps = 2 value heads x 4 diagonal blocks. Raw k k^T blocks shared by both heads.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define NW 8
struct KS5 {
  union { __nv_bfloat16 k[64 * 128]; struct { float raw[10][256]; float scr[6][256]; } p; } u;  // k: phase 1-2
  float aib[2][7][256];   // 0..3 diag Ai, 4 Ai10, 5 Ai21, 6 Ai20
  float g[2][64];
  float beta[2][64];
};

__device__ __forceinline__ int o128(int r, int c) { return r * 128 + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7); }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa(p)) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tf32(float (&c)[4], const float (&a)[4], float b0, float b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])), "r"(__float_as_uint(a[2])), "r"(__float_as_uint(a[3])),
                 "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)));
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

struct Blk { float e[2][4]; };  // C layout: e[t][q], t n-tile, q: (gq,2tq) (gq,2tq+1) (gq+8,2tq) (gq+8,2tq+1)
__device__ __forceinline__ void zero(Blk& b) {
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) b.e[t][q] = 0.f;
}
__device__ __forceinline__ void st_blk(float* M, const Blk& b, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    *reinterpret_cast<float2*>(&M[gq * 16 + 8 * t + 2 * tq]) = make_float2(b.e[t][0], b.e[t][1]);
    *reinterpret_cast<float2*>(&M[(gq + 8) * 16 + 8 * t + 2 * tq]) = make_float2(b.e[t][2], b.e[t][3]);
  }
}
__device__ __forceinline__ void ld_blk(Blk& b, const float* M, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const float2 x = *reinterpret_cast<const float2*>(&M[gq * 16 + 8 * t + 2 * tq]);
    const float2 y = *reinterpret_cast<const float2*>(&M[(gq + 8) * 16 + 8 * t + 2 * tq]);
    b.e[t][0] = x.x; b.e[t][1] = x.y; b.e[t][2] = y.x; b.e[t][3] = y.y;
  }
}
// acc += X @ Y with X, Y row-major 16x16 in smem; KW=1 natural k split, KW=2 even/odd k split across the two k8 steps
template <int KW>
__device__ __forceinline__ void mm_s(Blk& acc, const float* X, const float* Y, int lane) {
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    const int k0 = KW == 1 ? 8 * s + tq : 2 * tq + s;
    const int k1 = KW == 1 ? 8 * s + tq + 4 : 8 + 2 * tq + s;
    const float a[4] = {X[gq * 16 + k0], X[(gq + 8) * 16 + k0], X[gq * 16 + k1], X[(gq + 8) * 16 + k1]};
#pragma unroll
    for (int t = 0; t < 2; ++t) mma_tf32(acc.e[t], a, Y[k0 * 16 + 8 * t + gq], Y[k1 * 16 + 8 * t + gq]);
  }
}
__device__ __forceinline__ void negb(Blk& b) {
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int q = 0; q < 4; ++q) b.e[t][q] = __fsub_rn(0.f, b.e[t][q]);
}
__device__ __forceinline__ int bidx(int i, int j) { return i * (i + 1) / 2 + j; }

__global__ void __launch_bounds__(NW * 32, 3)
kkt7_kernel(const __nv_bfloat16* __restrict__ k, const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
            __nv_bfloat16* __restrict__ A, const int64_t* __restrict__ cu, const int64_t* __restrict__ cidx, int H, int HV) {
  extern __shared__ __align__(128) unsigned char smraw[];
  KS5& S = *reinterpret_cast<KS5*>(smraw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  const int i_n = (int)cidx[blockIdx.x * 2], i_t = (int)cidx[blockIdx.x * 2 + 1];
  const int kh = blockIdx.y;
  const int bos = (int)cu[i_n], T = (int)cu[i_n + 1] - bos;
  const int t0 = i_t * 64;
  const int nval = min(64, T - t0);
  // ---- phase 1: k chunk, g, beta ----
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int i = tid + 256 * e, r = i >> 4, c = i & 15;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < nval) val = *reinterpret_cast<const uint4*>(k + ((int64_t)(bos + t0 + r) * H + kh) * 128 + c * 8);
    *reinterpret_cast<uint4*>(&S.u.k[o128(r, c * 8)]) = val;
  }
  if (tid < 128) {
    const int v = tid >> 6, r = tid & 63, hv = kh * 2 + v;
    S.g[v][r] = r < nval ? g[(int64_t)(bos + t0 + r) * HV + hv] : 0.f;
  } else {
    const int v = (tid - 128) >> 6, r = (tid - 128) & 63, hv = kh * 2 + v;
    S.beta[v][r] = r < nval ? __bfloat162float(beta[(int64_t)(bos + t0 + r) * HV + hv]) : 0.f;
  }
  __syncthreads();
  // ---- phase 2: raw blocks (idx w and w+8) in registers; written over k after a barrier ----
  auto rawblk = [&](int idx) -> Blk {
    Blk c; zero(c);
    const int bi = (idx >= 1) + (idx >= 3) + (idx >= 6), bj = idx - bi * (bi + 1) / 2;
    if (16 * bi < nval) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t a[4], b[4];
        ldsm4(a, &S.u.k[o128(16 * bi + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
        ldsm4(b, &S.u.k[o128(16 * bj + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)]);
        mma_bf16(c.e[0], a, b[0], b[1]);
        mma_bf16(c.e[1], a, b[2], b[3]);
      }
    }
    return c;
  };
  const Blk rb0 = rawblk(warp);
  Blk rb1; zero(rb1);
  if (warp < 2) rb1 = rawblk(warp + 8);
  __syncthreads();  // all k reads done
  st_blk(S.u.p.raw[warp], rb0, lane);
  if (warp < 2) st_blk(S.u.p.raw[warp + 8], rb1, lane);
  __syncthreads();  // raw visible
  const int v = warp >> 2, b = warp & 3;
  const float* G = S.g[v];
  const float* Bt = S.beta[v];
  float* scr = S.u.p.scr[v * 3 + (b >= 1 ? b - 1 : 0)];
  // decayed/masked/beta-scaled block (bi, bj) for head v, from raw
  auto dec = [&](int bi, int bj) -> Blk {
    Blk x; ld_blk(x, S.u.p.raw[bidx(bi, bj)], lane);
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int lr = q < 2 ? gq : gq + 8, lc = 8 * t + 2 * tq + (q & 1);
        const int r = 16 * bi + lr, c = 16 * bj + lc;
        bool m = r < nval && c < nval;
        if (bi == bj) m = m && (lr > lc);
        const float e = m ? ex2(__fsub_rn(G[r], G[c])) : 0.f;
        x.e[t][q] = __fmul_rn(__fmul_rn(x.e[t][q], e), Bt[r]);
      }
    return x;
  };
  // ---- phase 3a: decayed diagonal blocks for both heads -> aib[v][0..3] ----
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int i = tid + 256 * e;
    const int hv_ = i >> 10, bb = (i >> 8) & 3, lr = (i >> 4) & 15, lc = i & 15;
    const int r = 16 * bb + lr, c = 16 * bb + lc;
    const bool m = r < nval && c < nval && lr > lc;
    const float e2 = m ? ex2(__fsub_rn(S.g[hv_][r], S.g[hv_][c])) : 0.f;
    S.aib[hv_][bb][lr * 16 + lc] = __fmul_rn(__fmul_rn(S.u.p.raw[bidx(bb, bb)][lr * 16 + lc], e2), S.beta[hv_][r]);
  }
  __syncthreads();
  // ---- phase 3b: column-per-thread forward substitution (16 threads per (head, block)) ----
  if (tid < 128) {
    const int P = tid >> 4, j = tid & 15, hv_ = P >> 2, bb = P & 3;
    float* D = S.aib[hv_][bb];
    float col[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) col[m] = __fsub_rn(0.f, D[m * 16 + j]);
    const int n_b = min(16, T - (t0 + 16 * bb));
#pragma unroll
    for (int r = 2; r < 16; ++r) {
      if (r < n_b) {
        const float* Dr = D + r * 16;
        auto exf = [&](int m) -> float { return (m < r) ? __fsub_rn(0.f, Dr[m]) : 0.f; };
        const float p0 = __fmaf_rn(col[0], exf(0), __fmul_rn(exf(8), col[8]));
        const float p4 = __fmaf_rn(col[4], exf(4), __fmul_rn(exf(12), col[12]));
        const float p2 = __fmaf_rn(col[2], exf(2), __fmul_rn(exf(10), col[10]));
        const float p6 = __fmaf_rn(col[6], exf(6), __fmul_rn(exf(14), col[14]));
        const float p1 = __fmaf_rn(col[1], exf(1), __fmul_rn(exf(9), col[9]));
        const float p5 = __fmaf_rn(col[5], exf(5), __fmul_rn(exf(13), col[13]));
        const float p3 = __fmaf_rn(col[3], exf(3), __fmul_rn(exf(11), col[11]));
        const float p7 = __fmaf_rn(col[7], exf(7), __fmul_rn(exf(15), col[15]));
        const float s = __fadd_rn(__fadd_rn(__fadd_rn(p0, p4), __fadd_rn(p2, p6)), __fadd_rn(__fadd_rn(p1, p5), __fadd_rn(p3, p7)));
        const float exj = (j < r) ? __fsub_rn(0.f, Dr[j]) : 0.f;
        col[r] = __fadd_rn(exj, s);
      }
    }
#pragma unroll
    for (int m = 0; m < 16; ++m) col[m] = __fadd_rn(col[m], m == j ? 1.f : 0.f);
    __syncwarp();   // all 16 threads of this problem done reading D before the in-place write
#pragma unroll
    for (int m = 0; m < 16; ++m) D[m * 16 + j] = col[m];
  }
  __syncthreads();
  float (*AIB)[256] = S.aib[v];
  // slots: 0..3 diag Ai, 4 Ai10, 5 Ai21, 6 Ai32, 7 Ai20, 8 Ai31, 9 Ai30
  const int hvv = kh * 2 + v;
  auto gstore = [&](const Blk& x, int bi, int bj) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int c = 16 * bj + 8 * t + 2 * tq;
      const int ra = 16 * bi + gq, rb2 = ra + 8;
      if (ra < nval) *reinterpret_cast<__nv_bfloat162*>(A + ((int64_t)(bos + t0 + ra) * HV + hvv) * 64 + c) = __floats2bfloat162_rn(x.e[t][0], x.e[t][1]);
      if (rb2 < nval) *reinterpret_cast<__nv_bfloat162*>(A + ((int64_t)(bos + t0 + rb2) * HV + hvv) * 64 + c) = __floats2bfloat162_rn(x.e[t][2], x.e[t][3]);
    }
  };
  // ---- phase 4, level 1: Ai10 (b=1), Ai21 (b=2), Ai32 (b=3) ----
  if (b >= 1) {
    const int i = b, j = b - 1;
    Blk Aij = dec(i, j);
    st_blk(scr, Aij, lane);
    __syncwarp();
    Blk tmp; zero(tmp); mm_s<2>(tmp, AIB[i], scr, lane);   // tl.dot(Ai_ii, A_ij)
    __syncwarp();
    st_blk(scr, tmp, lane);
    __syncwarp();
    Blk res; zero(res); mm_s<1>(res, scr, AIB[j], lane);   // tl.dot(tmp, Ai_jj)
    negb(res);
    if (b < 3) st_blk(AIB[4 + (b - 1)], res, lane); else gstore(res, 3, 2);
  }
  __syncthreads();
  // ---- level 2: Ai20 (b=2), Ai31 (b=3) ----
  if (b >= 2) {
    const int i = b;
    Blk acc; zero(acc);
    { Blk x = dec(i, i - 2); st_blk(scr, x, lane); __syncwarp(); mm_s<2>(acc, scr, AIB[i - 2], lane); __syncwarp(); }
    { Blk x = dec(i, i - 1); st_blk(scr, x, lane); __syncwarp(); mm_s<2>(acc, scr, AIB[b == 2 ? 4 : 5], lane); __syncwarp(); }
    st_blk(scr, acc, lane);
    __syncwarp();
    Blk res; zero(res); mm_s<1>(res, AIB[i], scr, lane);
    negb(res);
    if (b == 2) st_blk(AIB[6], res, lane); else gstore(res, 3, 1);
  }
  __syncthreads();
  // ---- level 3: Ai30 (b=3) ----
  if (b == 3) {
    Blk acc; zero(acc);
    { Blk x = dec(3, 0); st_blk(scr, x, lane); __syncwarp(); mm_s<2>(acc, scr, AIB[0], lane); __syncwarp(); }
    { Blk x = dec(3, 1); st_blk(scr, x, lane); __syncwarp(); mm_s<2>(acc, scr, AIB[4], lane); __syncwarp(); }
    { Blk x = dec(3, 2); st_blk(scr, x, lane); __syncwarp(); mm_s<2>(acc, scr, AIB[6], lane); __syncwarp(); }
    st_blk(scr, acc, lane);
    __syncwarp();
    Blk res; zero(res); mm_s<1>(res, AIB[3], scr, lane);
    negb(res);
    gstore(res, 3, 0);
  }
  __syncthreads();
  // ---- phase 5: stores of the blocks kept in smem (warp (v, b) stores row-block b) ----
  {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (j > b) break;
      const int d = b - j;
      int slot = -1;
      if (d == 0) slot = b;
      else if (d == 1 && b <= 2) slot = 3 + b;   // Ai10 -> 4, Ai21 -> 5
      else if (d == 2 && b == 2) slot = 6;        // Ai20
      if (slot < 0) continue;
      Blk x; ld_blk(x, AIB[slot], lane);
      gstore(x, b, j);
    }
  }
}

extern "C" int reax_kkt(const void* k, const void* g, const void* beta, void* A, const void* cu, const void* cidx, int NT, int H, int HV,
                    void* stream) {
  if (HV != 2 * H) return -1;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(kkt7_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(KS5)); init = true; }
  kkt7_kernel<<<dim3(NT, H), NW * 32, sizeof(KS5), (cudaStream_t)stream>>>((const __nv_bfloat16*)k, (const float*)g,
      (const __nv_bfloat16*)beta, (__nv_bfloat16*)A, (const int64_t*)cu, (const int64_t*)cidx, H, HV);
  return (int)cudaGetLastError();
}
extern "C" int reax_kkt_smem() { return (int)sizeof(KS5); }
