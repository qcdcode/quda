#pragma once

#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <array.h>
#include <shared_memory_cache_helper.h>
#include <kernel.h>
#include <warp_collective.h>
#include <dslash_quda.h>
#include <kernels/dslash_coarse.cuh>
#include <mma_tensor_op/gemm.cuh>

namespace quda {

  template <bool dslash_, bool clover_, bool dagger_, DslashType type_, typename Float,
            typename yFloat, typename ghostFloat, int nSpin_, int nColor_, int nVec_, int block_y_, int block_z_>
  struct DslashCoarseMmaArg : kernel_param<> {
    static constexpr bool dslash = dslash_;
    static constexpr bool clover = clover_;
    static constexpr bool dagger = dagger_;
    static constexpr DslashType type = type_;

    using real = typename mapper<Float>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nVec = nVec_;
    static constexpr int nDim = 4;
    static constexpr int nFace = 1;

    static constexpr int block_y = block_y_;
    static constexpr int block_z = block_z_;

    static constexpr int bM = nSpin * nColor;
    static constexpr int bN = nVec;
    static constexpr int bK = nSpin * nColor;

    static constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    static constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

    using ghost_accessor_t = typename colorspinor::GhostOrder<real, nSpin, nColor, nVec, csOrder, Float, ghostFloat>;
    // disable ghost to reduce arg size
    using field_accessor_t = typename colorspinor::FieldOrderCB<real, nSpin, nColor, nVec, csOrder, Float, ghostFloat, true>;
    using gauge_accessor_t = typename gauge::FieldOrder<real, nColor * nSpin, nSpin, gOrder, true, yFloat>;

    field_accessor_t out;
    const field_accessor_t inA;
    const field_accessor_t inB;
    const ghost_accessor_t halo;
    const gauge_accessor_t Y;
    const gauge_accessor_t X;

    const real kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;
    int ghostFaceCB[4];

    DslashCoarseMmaArg(ColorSpinorField &out, const ColorSpinorField &inA,
                    const ColorSpinorField &inB, const GaugeField &Y, const GaugeField &X,
                    real kappa, int parity, const ColorSpinorField &halo) :
      kernel_param(dim3(out.SiteSubset() * X.VolumeCB(), block_y, block_z)),
      out(out),
      inA(inA),
      inB(inB),
      halo(halo, nFace),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      X0h(((3 - nParity) * out.X(0)) / 2),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out.VolumeCB() / dim[4])
    {
      if (out.Nvec() != nVec) { errorQuda("out.Nvec() (%d) != nVec (%d)", out.Nvec(), nVec); }
      if (inA.Nvec() != nVec) { errorQuda("inA.Nvec() (%d) != nVec (%d)", inA.Nvec(), nVec); }
      if (inB.Nvec() != nVec) { errorQuda("inB.Nvec() (%d) != nVec (%d)", inB.Nvec(), nVec); }
      // ghostFaceCB does not include the batch (5th) dimension at present
      for (int i = 0; i < 4; i++) ghostFaceCB[i] = halo.getDslashConstant().ghostFaceCB[i];
    }
  };

  /**
     Applies the coarse dslash on a given parity and checkerboard site index
     /out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)

     @param[in,out] out The result vector
     @param[in] thread_dir Direction
     @param[in] x_cb The checkerboarded site index
     @param[in] src_idx Which src are we working on
     @param[in] parity The site parity
     @param[in] s_row Which spin row are acting on
     @param[in] color_block Which color row are we acting on
     @param[in] color_off Which color column offset are we acting on
     @param[in] arg Arguments
   */
  template <typename Arg>
  __device__ inline auto applyDslashMma(int x_cb, int parity, const Arg &arg)
  {
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    int coord[4];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);

    extern __shared__ half smem_ptr[];

    constexpr int M = Arg::nSpin * Arg::nColor;
    constexpr int N = Arg::nVec;
    constexpr int K = Arg::nSpin * Arg::nColor;

    constexpr int lda = M;
    constexpr int ldb = N;
    constexpr int ldc = N;

    using mma_t = mma::hmma_t;
    using Config = mma::MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

    constexpr int m_offset = 0;
    constexpr int n_offset = 0;

    static_assert(M <= Arg::bM, "Dividing M has NOT been implemented yet.\n");
    static_assert(N <= Arg::bN, "Dividing N has NOT been implemented yet.\n");
    static_assert(K <= Arg::bK, "Dividing K has NOT been implemented yet.\n");

    typename Config::SmemObjA smem_obj_a_real(smem_ptr);
    typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * Arg::bK);
    typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * Arg::bK);

    typename Config::Accumulator accumulator((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

    accumulator.zero();

    typename Config::ALoader a_loader;
    typename Config::BLoader b_loader;

    //Forward gather - compute fwd offset for spinor fetch
#pragma unroll
    for(int d = 0; d < Arg::nDim; d++) // loop over dimension
    {
      const int fwd_idx = linkIndexP1(coord, arg.dim, d);

      if (arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
        if constexpr (doHalo<Arg::type>()) {
#if 0
          int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
#pragma unroll
          for(int color_local = 0; color_local < Mc; color_local++) { //Color row
            int c_row = color_block + color_local; // global color index
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for(int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
              for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local] = cmac(arg.Y(d+4, parity, x_cb, row, col),
                      arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                else
                  out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                      arg.halo.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
              }
            }
          }
#endif
        }
      } else if constexpr (doBulk<Arg::type>()) {
#if 1
        auto a = arg.Y(Arg::dagger ? d : d + 4, parity, x_cb, 0, 0);
        auto b = arg.inA(their_spinor_parity, fwd_idx, 0, 0, 0);
        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;

        __syncthreads();
        a_loader.template g2r<lda, a_dagger>(a, m_offset, 0); // bk = 0
        a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);

        b_loader.template g2r<ldb, b_dagger>(b, n_offset, 0); // bk = 0
        b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();

        accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
#else
#pragma unroll
        for(int color_local = 0; color_local < Mc; color_local++) { //Color row
          int c_row = color_block + color_local; // global color index
          int row = s_row * Arg::nColor + c_row;
#pragma unroll
          for(int s_col = 0; s_col < Arg::nSpin; s_col++) { //Spin column
#pragma unroll
            for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) { //Color column
              int col = s_col * Arg::nColor + c_col + color_offset;
              if (!Arg::dagger)
                out[color_local] = cmac(arg.Y(d+4, parity, x_cb, row, col),
                    arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col+color_offset), out[color_local]);
              else
                out[color_local] = cmac(arg.Y(d, parity, x_cb, row, col),
                    arg.inA[src_idx](their_spinor_parity, fwd_idx, s_col, c_col+color_offset), out[color_local]);
            }
          }
        }
#endif
      }
    } // nDim

    //Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
    for(int d = 0; d < Arg::nDim; d++)
    {
      const int back_idx = linkIndexM1(coord, arg.dim, d);
      if (arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
        if constexpr (doHalo<Arg::type>()) {
#if 0
          const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
#pragma unroll
          for (int color_local=0; color_local<Mc; color_local++) {
            int c_row = color_block + color_local;
            int row = s_row * Arg::nColor + c_row;
#pragma unroll
            for (int s_col=0; s_col < Arg::nSpin; s_col++)
#pragma unroll
              for (int c_col=0; c_col < Arg::nColor; c_col += Arg::color_stride) {
                int col = s_col * Arg::nColor + c_col + color_offset;
                if (!Arg::dagger)
                  out[color_local] = cmac(conj(arg.Y.Ghost(d, 1-parity, ghost_idx, col, row)),
                      arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
                else
                  out[color_local] = cmac(conj(arg.Y.Ghost(d+4, 1-parity, ghost_idx, col, row)),
                      arg.halo.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx * arg.ghostFaceCB[d], s_col, c_col+color_offset), out[color_local]);
              }
          }
#endif
        }
      } else if constexpr (doBulk<Arg::type>()) {
        const int gauge_idx = back_idx;
#if 1
        auto a = arg.Y(Arg::dagger ? d + 4: d, 1 - parity, gauge_idx, 0, 0);
        auto b = arg.inA(their_spinor_parity, back_idx, 0, 0, 0);
        constexpr bool a_dagger = true;
        constexpr bool b_dagger = false;

        __syncthreads();
        a_loader.template g2r<lda, a_dagger>(a, m_offset, 0); // bk = 0
        a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);

        b_loader.template g2r<ldb, b_dagger>(b, n_offset, 0); // bk = 0
        b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();

        accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();
#else
#pragma unroll
        for(int color_local = 0; color_local < Mc; color_local++) {
          int c_row = color_block + color_local;
          int row = s_row * Arg::nColor + c_row;
#pragma unroll
          for(int s_col = 0; s_col < Arg::nSpin; s_col++)
#pragma unroll
            for(int c_col = 0; c_col < Arg::nColor; c_col += Arg::color_stride) {
              int col = s_col * Arg::nColor + c_col + color_offset;
              if (!Arg::dagger)
                out[color_local] = cmac(conj(arg.Y(d, 1-parity, gauge_idx, col, row)),
                    arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col+color_offset), out[color_local]);
              else
                out[color_local] = cmac(conj(arg.Y(d+4, 1-parity, gauge_idx, col, row)),
                    arg.inA[src_idx](their_spinor_parity, back_idx, s_col, c_col+color_offset), out[color_local]);
            }
        }
#endif
      }

    } //nDim

    return accumulator;
  }

  template <typename Arg> struct CoarseDslashMma {
    const Arg &arg;
    constexpr CoarseDslashMma(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ inline void operator()(int parity_x_cb, int thread_idx_y, int thread_idx_z)
    {
      int parity = (arg.nParity == 2) ? parity_x_cb % 2 : arg.parity;
      int x_cb = (arg.nParity == 2) ? parity_x_cb / 2 : parity_x_cb;

      int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
      auto out = applyDslashMma(x_cb, parity, arg);

      out.ax(-arg.kappa);

      // applyClover;

      constexpr int M = Arg::nSpin * Arg::nColor;
      constexpr int N = Arg::nVec;

      constexpr int ldc = N;

      if constexpr (doBulk<Arg::type>()) {
        auto c = arg.out(my_spinor_parity, x_cb, 0, 0);
        out.template store<M, N, ldc>(c, 0, 0);
      } else {
        // Do +=
      }
    }
  };

} // namespace quda
