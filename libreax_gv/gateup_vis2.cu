// Fused gate/up GEMM of the Qwen3.5 MLP with a compact activation output: CUTLASS's default sm80 mainloop (mma
// m16n8k16, fp32 accumulation, K ascending: the reference's sequential-K arithmetic, bit for bit) over the gate/up
// weights interleaved in groups of 4 rows ([g 4j..4j+3, u 4j..4j+3] for j = 0..N2/4-1), and an epilogue visitor:
// every 8-column accumulator vector holds 4 gate/up pairs of the same output features; the visitor rounds gate and
// up to bf16 (the reference's S=1 GEMM outputs), computes act = bf16(bf16(silu(g)) * u) with silu(g) = g / (1 + expf(-g))
// in fp32 (PyTorch's SiLU; no fast-math) and stores the 4 activations at column col/2 of the [M, N2] output.
// C API: reax_gv(cfg, x, w_g4, out, M, N2, K, stream) -> 0 on success; reax_gv_ncfg(), reax_gv_cfgname(cfg).
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using bf16 = cutlass::bfloat16_t;

template <class OutputTileIterator_>
struct SiLUMulVisitor {
  using OutputTileIterator = OutputTileIterator_;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static_assert(kElementsPerAccess == 8, "8-column accumulator vectors (4 gate + 4 up) expected");
  using ElementAccumulator = float;
  using AccumulatorFragment = cutlass::Array<float, kElementsPerAccess>;

  struct Params {
    typename OutputTileIterator::Params params_D;  // wide [M, 2*N2] layout: coordinates only, nothing stored through it
    bf16* out;
    int64_t ld_out;
  };
  struct SharedStorage {};

  Params const& params_;
  cutlass::MatrixCoord extent_;
  OutputTileIterator iterator_D_;

  CUTLASS_DEVICE
  SiLUMulVisitor(Params const& params, SharedStorage&, cutlass::MatrixCoord const& problem_size, int thread_idx, int,
                 int, cutlass::MatrixCoord const& threadblock_offset)
      : params_(params),
        extent_(problem_size),
        iterator_D_(params.params_D, nullptr, problem_size, thread_idx, threadblock_offset) {}

  CUTLASS_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE void set_batch_index(int) {}
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) {}
  CUTLASS_DEVICE void begin_row(int) {}

  CUTLASS_DEVICE
  void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
    cutlass::MatrixCoord c = iterator_D_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(frag_idx);
    if (c.row() < extent_.row() && c.column() < extent_.column()) {
      cutlass::Array<bf16, 4> r;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        float g = float(bf16(acc[i]));
        float u = float(bf16(acc[4 + i]));
        float s = float(bf16(g / (1.0f + expf(-g))));
        r[i] = bf16(s * u);
      }
      *reinterpret_cast<cutlass::Array<bf16, 4>*>(params_.out + int64_t(c.row()) * params_.ld_out + c.column() / 2) = r;
    }
  }

  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void end_step(int) { ++iterator_D_; }
  CUTLASS_DEVICE void end_epilogue() {}
};

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int S>
struct GV {
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using Base = typename cutlass::gemm::kernel::DefaultGemm<
      bf16, cutlass::layout::RowMajor, 8, bf16, cutlass::layout::ColumnMajor, 8, bf16, cutlass::layout::RowMajor,
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TBM, TBN, TBK>,
      cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>, Swizzle, S, false,
      cutlass::arch::OpMultiplyAdd>::GemmKernel;
  using Mma = typename Base::Mma;
  using Visitor = SiLUMulVisitor<typename Base::Epilogue::OutputTileIterator>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      Visitor, typename Base::Epilogue>::Epilogue;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    bf16 const* ptr_A;
    typename Mma::IteratorB::Params params_B;
    bf16 const* ptr_B;
    typename Visitor::Params visitor;
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename Visitor::SharedStorage visitor;
    } epilogue;
  };

  static __device__ void run(Params const& params, SharedStorage& shared_storage) {
    Swizzle swizzle;
    cutlass::gemm::GemmCoord tb = swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= tb.m() || params.grid_tiled_shape.n() <= tb.n()) return;
    int const K = params.problem_size.k();
    cutlass::MatrixCoord tb_offset_A{tb.m() * Mma::Shape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, tb.n() * Mma::Shape::kN};
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iterator_A(params.params_A, const_cast<bf16*>(params.ptr_A),
                                       {params.problem_size.m(), K}, thread_idx, tb_offset_A);
    typename Mma::IteratorB iterator_B(params.params_B, const_cast<bf16*>(params.ptr_B),
                                       {K, params.problem_size.n()}, thread_idx, tb_offset_B);
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int gemm_k_iterations = (K + Mma::Shape::kK - 1) / Mma::Shape::kK;
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    tb = swizzle.get_tile_offset(params.swizzle_log_tile);
    cutlass::MatrixCoord threadblock_offset(tb.m() * Mma::Shape::kM, tb.n() * Mma::Shape::kN);
    Visitor visitor(params.visitor, shared_storage.epilogue.visitor, params.problem_size.mn(), thread_idx, warp_idx,
                    lane_idx, threadblock_offset);
    Epilogue epilogue(shared_storage.epilogue.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(visitor, accumulators);
  }
};

template <class G>
__global__ void __launch_bounds__(G::Mma::WarpCount::kCount * 32) gv_kernel(typename G::Params params) {
  extern __shared__ int smem[];
  G::run(params, *reinterpret_cast<typename G::SharedStorage*>(smem));
}

template <class G>
static int launch(const void* x, const void* w, void* out, int M, int N2, int K, cudaStream_t stream) {
  if (K % 8 || N2 % 4) return 1;
  typename G::Params p;
  p.problem_size = cutlass::gemm::GemmCoord(M, 2 * N2, K);
  typename G::Swizzle swz;
  p.grid_tiled_shape = swz.get_tiled_shape(p.problem_size, {G::Mma::Shape::kM, G::Mma::Shape::kN, G::Mma::Shape::kK}, 1);
  p.swizzle_log_tile = swz.get_log_tile(p.grid_tiled_shape);
  p.params_A = typename G::Mma::IteratorA::Params(cutlass::layout::RowMajor(K));
  p.ptr_A = static_cast<bf16 const*>(x);
  p.params_B = typename G::Mma::IteratorB::Params(cutlass::layout::ColumnMajor(K));
  p.ptr_B = static_cast<bf16 const*>(w);
  p.visitor.params_D = typename G::Visitor::OutputTileIterator::Params(cutlass::layout::RowMajor(2 * N2));
  p.visitor.out = static_cast<bf16*>(out);
  p.visitor.ld_out = N2;
  dim3 grid = swz.get_grid_shape(p.grid_tiled_shape);
  dim3 block(G::Mma::WarpCount::kCount * 32, 1, 1);
  int smem = int(sizeof(typename G::SharedStorage));
  static bool attr = false;
  if (!attr) {
    if (smem >= (48 << 10) &&
        cudaFuncSetAttribute(gv_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
      return 2;
    attr = true;
  }
  gv_kernel<G><<<grid, block, smem, stream>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" {
int reax_gv_ncfg() { return 20; }
const char* reax_gv_cfgname(int c) {
  // 4..9: K-tile 64 with warp K 64 (one warp walks the whole K-tile, so each accumulator still adds its k16 mma
  // steps in ascending K order: the same bits as K-tile 32; checked at start-up anyway). <= 99 KB shared per block.
  static const char* n[] = {"128x256x32 w64x64 s3", "256x128x32 w64x64 s3", "128x128x32 w64x64 s4",
                            "128x256x32 w64x64 s4", "128x256x64 w64x64x64 s2", "256x128x64 w64x64x64 s2",
                            "128x128x64 w64x64x64 s3", "128x128x64 w64x64x64 s2", "64x256x64 w64x64x64 s2",
                            "128x256x32 w64x64 s2", "128x192x32 w64x48 s3", "128x192x32 w64x48 s4", "192x128x32 w48x64 s3", "192x128x32 w48x64 s4", "64x256x32 w64x64 s4", "64x256x32 w32x64 s4", "192x192x32 w96x48 s3", "128x128x32 w64x64 s3", "128x128x32 w64x32 s4", "256x128x32 w64x64 s4"};
  return (c >= 0 && c < 20) ? n[c] : "?";
}
// x [M, K] row-major, w_g4 [2*N2, K] row-major (gate/up rows interleaved in groups of 4), out [M, N2] row-major
int reax_gv(int cfg, const void* x, const void* w, void* out, int M, int N2, int K, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  switch (cfg) {
    case 0: return launch<GV<128, 256, 32, 64, 64, 32, 3>>(x, w, out, M, N2, K, s);
    case 1: return launch<GV<256, 128, 32, 64, 64, 32, 3>>(x, w, out, M, N2, K, s);
    case 2: return launch<GV<128, 128, 32, 64, 64, 32, 4>>(x, w, out, M, N2, K, s);
    case 3: return launch<GV<128, 256, 32, 64, 64, 32, 4>>(x, w, out, M, N2, K, s);
    case 4: return launch<GV<128, 256, 64, 64, 64, 64, 2>>(x, w, out, M, N2, K, s);
    case 5: return launch<GV<256, 128, 64, 64, 64, 64, 2>>(x, w, out, M, N2, K, s);
    case 6: return launch<GV<128, 128, 64, 64, 64, 64, 3>>(x, w, out, M, N2, K, s);
    case 7: return launch<GV<128, 128, 64, 64, 64, 64, 2>>(x, w, out, M, N2, K, s);
    case 8: return launch<GV<64, 256, 64, 64, 64, 64, 2>>(x, w, out, M, N2, K, s);
    case 9: return launch<GV<128, 256, 32, 64, 64, 32, 2>>(x, w, out, M, N2, K, s);
    case 10: return launch<GV<128, 192, 32, 64, 48, 32, 3>>(x, w, out, M, N2, K, s);
    case 11: return launch<GV<128, 192, 32, 64, 48, 32, 4>>(x, w, out, M, N2, K, s);
    case 12: return launch<GV<192, 128, 32, 48, 64, 32, 3>>(x, w, out, M, N2, K, s);
    case 13: return launch<GV<192, 128, 32, 48, 64, 32, 4>>(x, w, out, M, N2, K, s);
    case 14: return launch<GV<64, 256, 32, 64, 64, 32, 4>>(x, w, out, M, N2, K, s);
    case 15: return launch<GV<64, 256, 32, 32, 64, 32, 4>>(x, w, out, M, N2, K, s);
    case 16: return launch<GV<192, 192, 32, 96, 48, 32, 3>>(x, w, out, M, N2, K, s);
    case 17: return launch<GV<128, 128, 32, 64, 64, 32, 3>>(x, w, out, M, N2, K, s);
    case 18: return launch<GV<128, 128, 32, 64, 32, 32, 4>>(x, w, out, M, N2, K, s);
    case 19: return launch<GV<256, 128, 32, 64, 64, 32, 4>>(x, w, out, M, N2, K, s);
  }
  return 9;
}
}
