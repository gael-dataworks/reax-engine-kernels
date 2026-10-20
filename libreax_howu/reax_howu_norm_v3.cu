// v3 (v2 + branch-free exact SiLU in the gated-norm epilogue): exact CUDA port of gdn._howu (non-VCONV). One CTA per (seq, value head), 16 warps (<=128 regs).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BT 64
#define NTH 512

struct Smem {
  __nv_bfloat16 A[64 * 64];
  __nv_bfloat16 V[64 * 128];   // vb -> w -> vn
  __nv_bfloat16 K[64 * 128];
  __nv_bfloat16 Q[64 * 128];
  __nv_bfloat16 Hs[128 * 128]; // bf16(h) -> vdec (rows 0..63)
  __nv_bfloat16 AQ[64 * 64];
  float g[2][64];
  float eg[2][64];
  float beta[2][64];
};

__device__ __forceinline__ int o128(int r, int c) { return r * 128 + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7); }
__device__ __forceinline__ int o64(int r, int c) { return r * 64 + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7); }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa(p)) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa(p)) : "memory");
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cpa16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cpa_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float blo(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float bhi(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t mul2(uint32_t x, float s0, float s1) { return pk(__fmul_rn(blo(x), s0), __fmul_rn(bhi(x), s1)); }
__device__ __forceinline__ uint32_t kb2(uint32_t x, float b0, float b1, float e0, float e1) {
  uint32_t t = mul2(x, b0, b1);
  return pk(__fmul_rn(blo(t), e0), __fmul_rn(bhi(t), e1));
}
__device__ __forceinline__ void st32(__nv_bfloat16* p, uint32_t v) { *reinterpret_cast<uint32_t*>(p) = v; }

__device__ __forceinline__ float rcp_approx(float a) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d; }
// SiLU of a bf16 gate as the reference computes it: z / (1 + expf(-z)) (precise expf, div.rn)
__device__ __forceinline__ float silu_exact(float z) { return __fdiv_rn(z, __fadd_rn(1.f, expf(__fsub_rn(0.f, z)))); }
// same bits without div.rn's slow-path branch: refined reciprocal + one remainder correction, +-0 by select.
// Exhaustive check over all 65536 bf16 gates (rv2/cuda/silu_all_test.py): f32-bitwise equal to silu_exact for every
// finite gate > -87; others are flagged (fl) and recomputed with silu_exact.
__device__ __forceinline__ float silu_fast(float z, bool& fl) {
  const float d = __fadd_rn(1.f, expf(__fsub_rn(0.f, z)));
  float r = rcp_approx(d);
  r = __fmaf_rn(__fmaf_rn(-d, r, 1.f), r, r);
  float q = __fmul_rn(z, r);
  const float rem = __fmaf_rn(-d, q, z);
  q = __fmaf_rn(rem, r, q);
  q = z == 0.f ? z : q;
  fl |= !(z > -87.0f && fabsf(z) < INFINITY);
  return q;
}

template <bool SKIPA>
__global__ void __launch_bounds__(NTH, 1)
howu3p_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
             const __nv_bfloat16* __restrict__ beta, const __nv_bfloat16* __restrict__ Ag, const float* __restrict__ g,
             __nv_bfloat16* __restrict__ o, const int64_t* __restrict__ cu, float scale, int NSEQ, int H, int HV,
             const int* __restrict__ f2s, const __nv_bfloat16* __restrict__ zg, long long ZS, const __nv_bfloat16* __restrict__ nw,
             float eps, float factor) {
  extern __shared__ __align__(128) unsigned char smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int jw = warp >> 2;
  const int mt = ((warp & 3) + jw) & 3;   // t-row block, all 4 values per SMSP
  const int nc = jw;                      // 32-col block for t-work
  const int r0 = mt * 16, c0 = nc * 32;
  const int mk = warp >> 1, ch = warp & 1;  // h ownership: k rows 16mk.., cols 64ch..
  const int i_n = NSEQ - 1 - (int)(blockIdx.x / HV), hv = blockIdx.x % HV, kh = hv / (HV / H);
  const int bos = (int)cu[i_n], T = (int)cu[i_n + 1] - bos;
  const int NT = (T + BT - 1) / BT;

  auto issueA = [&](int t0, int nval) {
    const int r = tid >> 3, c = tid & 7;
    const bool ok = r < nval;
    const __nv_bfloat16* src = ok ? Ag + ((int64_t)(bos + t0 + r) * HV + hv) * BT + c * 8 : Ag;
    cpa16(&S.A[o64(r, c * 8)], src, ok);
  };
  auto issue128 = [&](__nv_bfloat16* dst, const __nv_bfloat16* base, int heads, int head, int t0, int nval) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int i = tid + NTH * e, r = i >> 4, c = i & 15;
      const bool ok = r < nval;
      const __nv_bfloat16* src = ok ? base + ((int64_t)(bos + t0 + r) * heads + head) * 128 + c * 8 : base;
      cpa16(&dst[o128(r, c * 8)], src, ok);
    }
  };
  auto prescaleV = [&](int slot) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int i = tid + NTH * e, r = i >> 4, c = i & 15;
      uint4* p = reinterpret_cast<uint4*>(&S.V[o128(r, c * 8)]);
      uint4 x = *p;
      const float b = S.beta[slot][r];
      x.x = mul2(x.x, b, b); x.y = mul2(x.y, b, b); x.z = mul2(x.z, b, b); x.w = mul2(x.w, b, b);
      *p = x;
    }
  };

  float h[8][4];  // k rows 16mk.., cols 64ch + 8j
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) h[j][e] = 0.f;
  for (int i = tid; i < 128 * 128 / 8; i += NTH) reinterpret_cast<uint4*>(S.Hs)[i] = make_uint4(0, 0, 0, 0);
  {
    const int nval = min(BT, T);
    issueA(0, nval);
    issue128(S.V, v, HV, hv, 0, nval);
    issue128(S.K, k, H, kh, 0, nval);
    issue128(S.Q, q, H, kh, 0, nval);
    cpa_commit();
    if (tid < BT) {
      float gg = 0.f, bb = 0.f;
      if (tid < nval) { gg = g[(int64_t)(bos + tid) * HV + hv]; bb = __bfloat162float(beta[(int64_t)(bos + tid) * HV + hv]); }
      S.g[0][tid] = gg; S.eg[0][tid] = ex2(gg); S.beta[0][tid] = bb;
    }
    cpa_wait_all();
    __syncthreads();
    prescaleV(0);
    __syncthreads();
  }

  for (int it = 0; it < NT; ++it) {
    const int t0 = it * BT, sl = it & 1;
    const int nvalid = min(BT, T - t0);
    const bool has_next = it + 1 < NT;
    const int t1 = t0 + BT, nval1 = has_next ? min(BT, T - t1) : 0;
    const float* G = S.g[sl];
    const int ta = r0 + gq, tb = ta + 8;
    // ---- P1+P2a: u = A @ vb, w = A @ kb (rows r0, cols c0..c0+31) ----
    float cu_[4][4], cw[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) { cu_[j][e] = 0.f; cw[j][e] = 0.f; }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      if (SKIPA && kk > mt) continue;
      uint32_t a[4], bv[2][4], bk[2][4];
      ldsm4(a, &S.A[o64(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        ldsm4t(bv[jj], &S.V[o128(kk * 16 + (lane & 15), c0 + jj * 16 + (lane >> 4) * 8)]);
        ldsm4t(bk[jj], &S.K[o128(kk * 16 + (lane & 15), c0 + jj * 16 + (lane >> 4) * 8)]);
      }
      const int s = kk * 16 + 2 * tq;
      const float b0 = S.beta[sl][s], b1 = S.beta[sl][s + 1], b2 = S.beta[sl][s + 8], b3 = S.beta[sl][s + 9];
      const float e0 = S.eg[sl][s], e1 = S.eg[sl][s + 1], e2 = S.eg[sl][s + 8], e3 = S.eg[sl][s + 9];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        mma(cu_[2 * jj], a, bv[jj][0], bv[jj][1]);
        mma(cu_[2 * jj + 1], a, bv[jj][2], bv[jj][3]);
        const uint32_t x0 = kb2(bk[jj][0], b0, b1, e0, e1), x1 = kb2(bk[jj][1], b2, b3, e2, e3);
        const uint32_t x2 = kb2(bk[jj][2], b0, b1, e0, e1), x3 = kb2(bk[jj][3], b2, b3, e2, e3);
        mma(cw[2 * jj], a, x0, x1);
        mma(cw[2 * jj + 1], a, x2, x3);
      }
    }
    uint32_t ub[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) { ub[j][0] = pk(cu_[j][0], cu_[j][1]); ub[j][1] = pk(cu_[j][2], cu_[j][3]); }
    // ---- P2b: Aq = q @ k^T (rows r0, s tiles 2nc, 2nc+1), decay+mask -> AQ ----
    {
      float c[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[j][e] = 0.f;
      const int jlim = 2 * mt + 1, jg0 = 2 * nc;
      if (jg0 <= jlim) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          uint32_t a[4], b[4];
          ldsm4(a, &S.Q[o128(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
          ldsm4(b, &S.K[o128(nc * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)]);
          mma(c[0], a, b[0], b[1]);
          if (jg0 + 1 <= jlim) mma(c[1], a, b[2], b[3]);
        }
      }
      const float ga = G[ta], gb = G[tb];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int jg = jg0 + j;
        const int s0 = 8 * jg + 2 * tq, s1 = s0 + 1;
        float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
        if (jg <= jlim) {
          const float gs0 = G[s0], gs1 = G[s1];
          const bool va = ta < nvalid, vb = tb < nvalid, v0 = s0 < nvalid, v1 = s1 < nvalid;
          const float z0 = __fmul_rn(c[j][0], ex2(__fsub_rn(ga, gs0)));
          const float z1 = __fmul_rn(c[j][1], ex2(__fsub_rn(ga, gs1)));
          const float z2 = __fmul_rn(c[j][2], ex2(__fsub_rn(gb, gs0)));
          const float z3 = __fmul_rn(c[j][3], ex2(__fsub_rn(gb, gs1)));
          y0 = (ta >= s0 && va && v0) ? z0 : 0.f;
          y1 = (ta >= s1 && va && v1) ? z1 : 0.f;
          y2 = (tb >= s0 && vb && v0) ? z2 : 0.f;
          y3 = (tb >= s1 && vb && v1) ? z3 : 0.f;
        }
        st32(&S.AQ[o64(ta, s0)], pk(y0, y1));
        st32(&S.AQ[o64(tb, s0)], pk(y2, y3));
      }
    }
    float ng = 0.f, nb = 0.f;
    if (has_next && tid < BT && tid < nval1) {
      ng = g[(int64_t)(bos + t1 + tid) * HV + hv];
      nb = __bfloat162float(beta[(int64_t)(bos + t1 + tid) * HV + hv]);
    }
    __syncthreads();  // S1: A, V(vb) reads done
    if (has_next) { issueA(t1, nval1); cpa_commit(); }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = c0 + 8 * j + 2 * tq;
      st32(&S.V[o128(ta, col)], pk(cw[j][0], cw[j][1]));
      st32(&S.V[o128(tb, col)], pk(cw[j][2], cw[j][3]));
    }
    __syncthreads();  // S2: w visible
    // ---- P3: vw = w @ h ; oi = q @ h ----
    float vw[4][4], oi[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) { vw[j][e] = 0.f; oi[j][e] = 0.f; }
    if (it > 0) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t aw[4], aq[4], b[2][4];
        ldsm4(aw, &S.V[o128(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
        ldsm4(aq, &S.Q[o128(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) ldsm4t(b[jj], &S.Hs[o128(kk * 16 + (lane & 15), c0 + jj * 16 + (lane >> 4) * 8)]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          mma(vw[2 * jj], aw, b[jj][0], b[jj][1]);
          mma(vw[2 * jj + 1], aw, b[jj][2], b[jj][3]);
          mma(oi[2 * jj], aq, b[jj][0], b[jj][1]);
          mma(oi[2 * jj + 1], aq, b[jj][2], b[jj][3]);
        }
      }
    }
    const float glast = G[nvalid - 1];
    const float da = ta < nvalid ? ex2(__fsub_rn(glast, G[ta])) : 0.f;
    const float db = tb < nvalid ? ex2(__fsub_rn(glast, G[tb])) : 0.f;
    uint32_t vnp[4][2], vdp[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x0 = __fsub_rn(blo(ub[j][0]), vw[j][0]);
      const float x1 = __fsub_rn(bhi(ub[j][0]), vw[j][1]);
      const float x2 = __fsub_rn(blo(ub[j][1]), vw[j][2]);
      const float x3 = __fsub_rn(bhi(ub[j][1]), vw[j][3]);
      vnp[j][0] = pk(x0, x1); vnp[j][1] = pk(x2, x3);
      vdp[j][0] = pk(__fmul_rn(x0, da), __fmul_rn(x1, da));
      vdp[j][1] = pk(__fmul_rn(x2, db), __fmul_rn(x3, db));
    }
    __syncthreads();  // S3: V(w), Q, Hs reads done
    int srw[4];
    uint2 zpf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = warp + 16 * i;
      srw[i] = t < nvalid ? f2s[bos + t0 + t] : -1;
      zpf[i] = srw[i] >= 0 ? *reinterpret_cast<const uint2*>(zg + (int64_t)srw[i] * ZS + hv * 128 + 4 * lane) : make_uint2(0, 0);
    }
    if (has_next) { issue128(S.Q, q, H, kh, t1, nval1); cpa_commit(); }
    if (tid < BT) { S.g[sl ^ 1][tid] = ng; S.eg[sl ^ 1][tid] = ex2(ng); S.beta[sl ^ 1][tid] = nb; }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = c0 + 8 * j + 2 * tq;
      st32(&S.V[o128(ta, col)], vnp[j][0]);
      st32(&S.V[o128(tb, col)], vnp[j][1]);
      st32(&S.Hs[o128(ta, col)], vdp[j][0]);
      st32(&S.Hs[o128(tb, col)], vdp[j][1]);
    }
    __syncthreads();  // S4: vn, vdec visible
    // ---- P5a: o_intra = Aq @ vn ; o = fma(o_intra, scale, (oi*exp2(g))*scale) ----
    {
      float ci[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) ci[j][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        if (kk > mt) continue;
        uint32_t a[4], b[2][4];
        ldsm4(a, &S.AQ[o64(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) ldsm4t(b[jj], &S.V[o128(kk * 16 + (lane & 15), c0 + jj * 16 + (lane >> 4) * 8)]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) { mma(ci[2 * jj], a, b[jj][0], b[jj][1]); mma(ci[2 * jj + 1], a, b[jj][2], b[jj][3]); }
      }
      const float ea = ex2(G[ta]), eb = ex2(G[tb]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float r[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) r[e] = __fmaf_rn(ci[j][e], scale, __fmul_rn(__fmul_rn(oi[j][e], e < 2 ? ea : eb), scale));
        const int col = c0 + 8 * j + 2 * tq;
        st32(&S.Hs[o128(64 + ta, col)], pk(r[0], r[1]));
        st32(&S.Hs[o128(64 + tb, col)], pk(r[2], r[3]));
      }
    }
    // ---- P5b: h = h*exp2(glast) + k^T @ vdec (k rows 16mk.., cols 64ch..) ----
    {
      const float gl = ex2(glast);
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) h[j][e] = __fmul_rn(h[j][e], gl);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t a[4], b[4][4];
        const int sr = kk * 16 + (lane & 7) + ((lane >> 4) << 3), sc = ((lane >> 3) & 1) * 8;
        ldsm4t(a, &S.K[o128(sr, mk * 16 + sc)]);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) ldsm4t(b[jj], &S.Hs[o128(kk * 16 + (lane & 15), ch * 64 + jj * 16 + (lane >> 4) * 8)]);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { mma(h[2 * jj], a, b[jj][0], b[jj][1]); mma(h[2 * jj + 1], a, b[jj][2], b[jj][3]); }
      }
    }
    __syncthreads();  // S5: all reads of V, K, Hs(vdec), AQ done
    if (has_next) {
      issue128(S.V, v, HV, hv, t1, nval1);
      issue128(S.K, k, H, kh, t1, nval1);
      cpa_commit();
    }
    // fused gated RMSNorm epilogue (exact _gated_rmsnorm semantics); o (bf16) is staged in Hs rows 64..127
    auto norm_row = [&](int i, bool exact, bool& fl) {
      const int t = warp + 16 * i;
      const int srow = srw[i];
      const uint2 rw = *reinterpret_cast<const uint2*>(&S.Hs[o128(64 + t, 4 * lane)]);
      const float x[4] = {blo(rw.x), bhi(rw.x), blo(rw.y), bhi(rw.y)};
      float p = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[1], x[1])), __fmul_rn(x[2], x[2])), __fmul_rn(x[3], x[3]));
      p = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 16));
      p = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 8));
      p = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 4));
      p = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 2));
      p = __fadd_rn(p, __shfl_down_sync(0xffffffffu, p, 1));
      const float sum = __shfl_sync(0xffffffffu, p, 0);
      const float rr = rsqrtf(__fadd_rn(__fmul_rn(sum, factor), eps));
      const uint2 wv = *reinterpret_cast<const uint2*>(nw + 4 * lane);
      const uint2 zv = zpf[i];
      const float w[4] = {blo(wv.x), bhi(wv.x), blo(wv.y), bhi(wv.y)};
      const float zz[4] = {blo(zv.x), bhi(zv.x), blo(zv.y), bhi(zv.y)};
      float res[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float y = bfr(__fmul_rn(x[e], rr));
        const float y2 = bfr(__fmul_rn(w[e], y));
        const float sz = exact ? silu_exact(zz[e]) : silu_fast(zz[e], fl);
        res[e] = __fmul_rn(y2, sz);
      }
      *reinterpret_cast<uint2*>(o + ((int64_t)srow * HV + hv) * 128 + 4 * lane) = make_uint2(pk(res[0], res[1]), pk(res[2], res[3]));
    };
    uint32_t wmask = 0;  // warp-uniform: rows with a gate the fast SiLU does not cover (<= -87, inf, NaN)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (srw[i] >= 0) {
        bool fl = false;
        norm_row(i, false, fl);
        wmask |= __any_sync(0xffffffffu, fl) ? (1u << i) : 0u;
      }
    }
    if (wmask) {  // never taken for finite gates > -87
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        bool fl = false;
        if ((wmask >> i) & 1u) norm_row(i, true, fl);
      }
    }
    if (mk < 4) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = ch * 64 + 8 * j + 2 * tq;
        st32(&S.Hs[o128(mk * 16 + gq, col)], pk(h[j][0], h[j][1]));
        st32(&S.Hs[o128(mk * 16 + gq + 8, col)], pk(h[j][2], h[j][3]));
      }
    }
    __syncthreads();  // S5b: o copy-out reads of Hs[64..127] done
    if (mk >= 4) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = ch * 64 + 8 * j + 2 * tq;
        st32(&S.Hs[o128(mk * 16 + gq, col)], pk(h[j][0], h[j][1]));
        st32(&S.Hs[o128(mk * 16 + gq + 8, col)], pk(h[j][2], h[j][3]));
      }
    }
    if (has_next) {
      cpa_wait_all();
      prescaleV(sl ^ 1);
    }
    __syncthreads();  // S0
  }
}

template <bool SKIPA>
static int launch(const void* q, const void* k, const void* v, const void* beta, const void* A, const void* g, void* o,
                  const void* cu, float scale, int N, int H, int HV, void* stream, const void* f2s, const void* z, long long ZS,
                  const void* nw, float eps, float factor) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(howu3p_kernel<SKIPA>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
    init = true;
  }
  howu3p_kernel<SKIPA><<<N * HV, NTH, sizeof(Smem), (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, (const __nv_bfloat16*)beta,
      (const __nv_bfloat16*)A, (const float*)g, (__nv_bfloat16*)o, (const int64_t*)cu, scale, N, H, HV,
      (const int*)f2s, (const __nv_bfloat16*)z, ZS, (const __nv_bfloat16*)nw, eps, factor);
  return (int)cudaGetLastError();
}
extern "C" int reax_howu_norm(const void* q, const void* k, const void* v, const void* beta, const void* A, const void* g,
                              void* out, const void* cu, float scale, int N, int H, int HV, void* stream, const void* f2s,
                              const void* z, long long ZS, const void* nw, float eps, float factor) {
  return launch<true>(q, k, v, beta, A, g, out, cu, scale, N, H, HV, stream, f2s, z, ZS, nw, eps, factor);
}
