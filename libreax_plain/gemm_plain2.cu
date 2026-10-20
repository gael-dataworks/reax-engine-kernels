// Plain bf16 GEMM D[M, N] = X[M, K] @ W[N, K]^T with CUTLASS's sm80 kernels (mma m16n8k16, fp32 accumulation, K
// ascending: the reference's sequential-K arithmetic), bf16 output rounded once.
// C API: reax_gemm(cfg, x, w, d, M, N, K, ldd, stream) -> 0 on success; reax_gemm_ncfg(), reax_gemm_cfgname(cfg).
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using bf16 = cutlass::bfloat16_t;

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int S>
using G = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, bf16, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, S>;

template <class Gemm>
static int run(const void* x, const void* w, void* d, int M, int N, int K, int ldd, cudaStream_t stream) {
  typename Gemm::Arguments args({M, N, K}, {(bf16 const*)x, K}, {(bf16 const*)w, K}, {(bf16 const*)d, ldd},
                                {(bf16*)d, ldd}, {1.0f, 0.0f});
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 2;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return 3;
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" {
int reax_gemm_ncfg() { return 15; }
const char* reax_gemm_cfgname(int c) {
  static const char* n[] = {"128x256x32 w64x64 s3", "256x128x32 w64x64 s3", "128x128x32 w64x64 s4", "128x256x32 w64x64 s4", "128x256x32 w64x64 s2", "192x128x32 w48x64 s3", "192x128x32 w48x64 s4", "64x256x32 w64x64 s4", "64x256x32 w32x64 s4", "128x128x32 w64x64 s3", "128x128x32 w64x32 s4", "256x128x32 w64x64 s4", "256x64x32 w64x64 s4", "128x64x32 w64x64 s4", "64x128x32 w64x64 s4"};
  return (c >= 0 && c < 15) ? n[c] : "?";
}
int reax_gemm(int cfg, const void* x, const void* w, void* d, int M, int N, int K, int ldd, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  switch (cfg) {
    case 0: return run<G<128, 256, 32, 64, 64, 32, 3>>(x, w, d, M, N, K, ldd, s);
    case 1: return run<G<256, 128, 32, 64, 64, 32, 3>>(x, w, d, M, N, K, ldd, s);
    case 2: return run<G<128, 128, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 3: return run<G<128, 256, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 4: return run<G<128, 256, 32, 64, 64, 32, 2>>(x, w, d, M, N, K, ldd, s);
    case 5: return run<G<192, 128, 32, 48, 64, 32, 3>>(x, w, d, M, N, K, ldd, s);
    case 6: return run<G<192, 128, 32, 48, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 7: return run<G<64, 256, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 8: return run<G<64, 256, 32, 32, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 9: return run<G<128, 128, 32, 64, 64, 32, 3>>(x, w, d, M, N, K, ldd, s);
    case 10: return run<G<128, 128, 32, 64, 32, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 11: return run<G<256, 128, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 12: return run<G<256, 64, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 13: return run<G<128, 64, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
    case 14: return run<G<64, 128, 32, 64, 64, 32, 4>>(x, w, d, M, N, K, ldd, s);
  }
  return 9;
}
}
