// Split-K partials as one batched GEMM: batch s computes P[s] = X[:, s*kps:(s+1)*kps] @ W[:, s*kps:(s+1)*kps]^T with
// CUTLASS's sm80 kernels (mma m16n8k16, fp32 accumulation, K ascending within the slice: the reference's per-slice
// chain), each partial rounded once to its storage type (bf16 or fp32), exactly as exact_gemm._gemm's partial modes.
// C API: reax_slices(cfg, part_fp32, x, w, p, M, N, kps, S, K, stream) -> 0 on success.
// NOTE on exactness: every config keeps threadblock-K = warp-K (32 or 64: one warp walks the whole K-tile) and the
// m16n8k16 mma atom, so the K-reduction
// order (hence the fp32 rounding) is identical across configs -- only tile M/N, warp M/N, stages and swizzle vary,
// none of which touch the reduction. Each config is still bit-checked at startup before use.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using bf16 = cutlass::bfloat16_t;

template <class Out, int TBM, int TBN, int TBK, int WM, int WN, int WK, int St>
using GB = cutlass::gemm::device::GemmBatched<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, Out, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Out, 128 / cutlass::sizeof_bits<Out>::value, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, St>;

template <class G, class Out>
static int run(const void* x, const void* w, void* p, int M, int N, int kps, int S, int K, cudaStream_t stream) {
  typename G::Arguments args({M, N, kps}, {(bf16 const*)x, K}, (long long)kps, {(bf16 const*)w, K}, (long long)kps,
                             {(Out const*)p, N}, (long long)M * N, {(Out*)p, N}, (long long)M * N, {1.0f, 0.0f}, S);
  G g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (g.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 2;
  if (g.run(stream) != cutlass::Status::kSuccess) return 3;
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

// cfg -> (Out=float) and (Out=bf16) instantiation. All keep TBK=WK=32 (bit-exact; see header note).
#define SLICE_CFGS(X)                         \
  X(0, 128, 256, 32, 64, 64, 32, 3)           \
  X(1, 256, 128, 32, 64, 64, 32, 3)           \
  X(2, 128, 128, 32, 64, 64, 32, 4)           \
  X(3, 64, 256, 32, 64, 64, 32, 4)            \
  X(4, 128, 128, 32, 64, 64, 32, 3)           \
  X(5, 256, 128, 32, 64, 64, 32, 4)           \
  X(6, 128, 256, 32, 64, 64, 32, 4)           \
  X(7, 64, 128, 32, 64, 64, 32, 4)            \
  X(8, 128, 64, 32, 64, 64, 32, 4)            \
  X(9, 64, 256, 32, 64, 64, 32, 3)            \
  X(10, 64, 128, 32, 32, 64, 32, 4)           \
  X(11, 256, 64, 32, 64, 64, 32, 3)           \
  X(12, 128, 128, 32, 64, 64, 32, 5)          \
  X(13, 128, 128, 32, 64, 64, 32, 6)          \
  X(14, 64, 128, 32, 32, 64, 32, 6)           \
  X(15, 256, 128, 32, 64, 64, 32, 5)

extern "C" {
int reax_slices_ncfg() { return 16; }
const char* reax_slices_cfgname(int c) {
  static const char* n[] = {
      "128x256x32 w64x64 s3", "256x128x32 w64x64 s3", "128x128x32 w64x64 s4", "64x256x32 w64x64 s4",
      "128x128x32 w64x64 s3", "256x128x32 w64x64 s4", "128x256x32 w64x64 s4", "64x128x32 w64x64 s4",
      "128x64x32 w64x64 s4",  "64x256x32 w64x64 s3",  "64x128x32 w32x64 s4",  "256x64x32 w64x64 s3",
      "128x128x32 w64x64 s5", "128x128x32 w64x64 s6", "64x128x32 w32x64 s6",  "256x128x32 w64x64 s5"};
  return (c >= 0 && c < 16) ? n[c] : "?";
}
int reax_slices(int cfg, int part_fp32, const void* x, const void* w, void* p, int M, int N, int kps, int S, int K,
                void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (part_fp32) {
    switch (cfg) {
#define X(id, TBM, TBN, TBK, WM, WN, WK, St) \
  case id: return run<GB<float, TBM, TBN, TBK, WM, WN, WK, St>, float>(x, w, p, M, N, kps, S, K, s);
      SLICE_CFGS(X)
#undef X
    }
  } else {
    switch (cfg) {
#define X(id, TBM, TBN, TBK, WM, WN, WK, St) \
  case id: return run<GB<bf16, TBM, TBN, TBK, WM, WN, WK, St>, bf16>(x, w, p, M, N, kps, S, K, s);
      SLICE_CFGS(X)
#undef X
    }
  }
  return 9;
}
}
