#pragma once

#include <complex_quda.h>
#include <quda_matrix.h>

/**
 * @file    gen_matrix.h
 *
 * @section Description
 *
 * The header file defines some helper structs for dealing with
 * matrices with M rows and N columns in column-major storage
 * for easy compatibility with CuBlas. 
 */
namespace quda {

  template<typename Float, typename T> struct gen_matrix_wrapper;
  template<typename Float, typename T> struct gen_matrix_ghost_wrapper;

  /**
     This is the generic declaration of GenMatrix.
     @tparam M number of rows
     @tparam N number of columns
   */
  template <typename Float, int M, int N>
    struct GenMatrix {

    static constexpr int size = M * N;
    complex<Float> data[size];

    __device__ __host__ inline GenMatrix<Float, M, N>()
    {
#pragma unroll
      for (int i = 0; i < size; i++) { data[i] = 0; }
      }

      __device__ __host__ inline GenMatrix<Float, M, N>(const GenMatrix<Float, M, N> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }

      __device__ __host__ inline GenMatrix<Float, M, N>& operator=(const GenMatrix<Float, M, N> &a) {
	if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
        }
	return *this;
      }

      __device__ __host__ inline GenMatrix<Float, M, N> operator-() const
      {
        GenMatrix<Float, M, N> a;
#pragma unroll
        for (int i = 0; i < size; i++) { a.data[i] = -data[i]; }
        return a;
      }

      __device__ __host__ inline GenMatrix<Float, M, N>& operator+=(const GenMatrix<Float, M, N> &a) {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] += a.data[i]; }
        return *this;
      }

      template <typename T> __device__ __host__ inline GenMatrix<Float, M, N> &operator*=(const T &a)
      {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] *= a; }
        return *this;
      }

      __device__ __host__ inline GenMatrix<Float, M, N> &operator-=(const GenMatrix<Float, M, N> &a)
      {
        if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] -= a.data[i]; }
        }
        return *this;
      }

      template<typename S>
      __device__ __host__ inline GenMatrix<Float, M, N>(const gen_matrix_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const gen_matrix_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline GenMatrix<Float, M, N>(const gen_matrix_ghost_wrapper<Float, S> &s);

      template<typename S>
      __device__ __host__ inline void operator=(const gen_matrix_ghost_wrapper<Float, S> &s);

      /**
        @brief 2-d accessor functor
        @param[in] r Row index
        @param[in] c Column index
        @return Complex number at this row and column index
      */
      __device__ __host__ inline complex<Float>& operator()(int r, int c) { return data[c*M + r]; }

      /**
        @brief 2-d accessor functor
        @param[in] r Row index
        @param[in] c Column index
        @return Complex number at this row and column index
      */
      __device__ __host__ inline const complex<Float>& operator()(int r, int c) const { return data[c*M + r]; }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline complex<Float>& operator()(int idx) { return data[idx]; }

      /**
         @brief 1-d accessor functor
         @param[in] idx Index
         @return Complex number at this index
      */
      __device__ __host__ inline const complex<Float>& operator()(int idx) const { return data[idx]; }

      /**
         @brief Prints the NxM complex elements of the matrix
      */
      __device__ __host__ void print() const
      {
        for (int r=0; r<M; r++) {
          for (int c=0; c<N; c++) {
            printf("r=%d c=%d %e %e\n", r, c, data[c*M+r].real(), data[c*M+r].imag());
          }
        }
      }
    };
  
  /**
     @brief Compute y = a * x + y
     @param a Scaling factor
     @param x GenMatrix to be scaled and added
     @param y GenMatrix to be accumulated onto
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline void caxpy(const complex<Float> &a, const GenMatrix<Float, M, N> &x,
                                        GenMatrix<Float, M, N> &y)
  {
#pragma unroll
    for (int i = 0; i < M * N; i++) {
      y(i).real( a.real() * x(i).real() + y(i).real());
      y(i).real(-a.imag() * x(i).imag() + y(i).real());
      y(i).imag( a.imag() * x(i).real() + y(i).imag());
      y(i).imag( a.real() * x(i).imag() + y(i).imag());
    }
  }

  /**
     @brief Compute the L2 norm squared over column and row
     nrm = \sum_r,c conj(a(r,c)) * a(r,c)
     @param a GenMatrix we taking the norm of
     @return The L2 norm squared
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline Float norm2(const GenMatrix<Float, M, N> &a)
  {
    Float nrm = 0.0;
#pragma unroll
    for (int i = 0; i < M * N; i++) {
      nrm += a(i).real() * a(i).real();
      nrm += a(i).imag() * a(i).imag();
    }
    return nrm;
  }


  /**
     @brief Compute the inner product over column and row
     dot = \sum_r,c conj(a(r,c)) * b(r,c)
     @param a Left-hand side GenMatrix
     @param b Right-hand side GenMatrix
     @return The inner product
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline complex<Float> innerProduct(const GenMatrix<Float, M, N> &a,
                                                         const GenMatrix<Float, M, N> &b)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int r = 0; r < M; r++) { dot += innerProduct(a, b, r, r); }
    return dot;
  }

  /**
     @brief Compute the column contraction at rows ra and rb
     dot = \sum_c a(ra,c) * b(rb,c)
     @param a Left-hand side GenMatrix
     @param b Right-hand side GenMatrix
     @param ra Left-hand side row index
     @param rb Right-hand side row index
     @return The column contraction
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline complex<Float> columnContract(const GenMatrix<Float, M, N> &a,
                                                           const GenMatrix<Float, M, N> &b, int ra, int rb)
  {
    complex<Float> dot = 0;
    for (int c = 0; c < N, c++) {
      dot.real(dot.real() + a(ra, c).real() * b(rb, c).real());
      dot.real(dot.real() - a(ra, c).imag() * b(rb, c).imag());
      dot.imag(dot.imag() + a(ra, c).real() * b(rb, c).imag());
      dot.imag(dot.imag() + a(ra, c).imag() * b(rb, c).real());
    }

    return dot;
  }

  /**
     Compute the inner product over column at row r between two GenMatrix matrices of the same size
     dot = \sum_c conj(a(r,c)) * b(r,c)
     @param a Left-hand side GenMatrix
     @param b Right-hand side GenMatrix
     @param r diagonal row index
     @return The inner product
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline complex<Float> innerProduct(const GenMatrix<Float, M, N> &a,
                                                         const GenMatrix<Float, M, N> &b, int r)
  {
    return innerProduct(a, b, r, r);
  }

  /**
     Compute the inner product over columns at row ra and rb  between two GenMatrix matrices
     dot = \sum_c conj(a(ra,c)) * b(rb,c)
     @param a Left-hand side GenMatrix
     @param b Right-hand side GenMatrix
     @param ra Left-hand side row index
     @param rb Right-hand side row index
     @return The inner product
  */
  template <typename Float, int M, int N>
  __device__ __host__ inline complex<Float> innerProduct(const GenMatrix<Float, M, N> &a,
                                                         const GenMatrix<Float, M, N> &b, int ra, int rb)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int c = 0; c < N; c++) {
      dot.real(dot.real() + a(ra, c).real() * b(rb, c).real());
      dot.real(dot.real() + a(ra, c).imag() * b(rb, c).imag());
      dot.imag(dot.imag() + a(ra, c).real() * b(rb, c).imag());
      dot.imag(dot.imag() - a(ra, c).imag() * b(rb, c).real());
    }
    return dot;
  }

  /**
     @brief Compute the inner product over column at row ra and rb between a
     a and b with different numbers of rows but same number of columns
     dot = \sum_c conj(a(ra,c)) * b(rb,c)
     @param a Left-hand side GenMatrix
     @param b Right-hand side GenMatrix
     @param ra Left-hand side row index
     @param rb Right-hand side row index
     @return The inner product
  */
  template <typename Float, int Ma, int Mb, int N>
  __device__ __host__ inline complex<Float> innerProduct(const GenMatrix<Float, Ma, N> &a,
                                                         const GenMatrix<Float, Mb, N> &b, int ra, int rb)
  {
    complex<Float> dot = 0;
#pragma unroll
    for (int c = 0; c < N; c++) {
      dot.real(dot.real() + a(ra, c).real() * b(rb, c).real());
      dot.real(dot.real() + a(ra, c).imag() * b(rb, c).imag());
      dot.imag(dot.imag() + a(ra, c).real() * b(rb, c).imag());
      dot.imag(dot.imag() - a(ra, c).imag() * b(rb, c).real());
    }
    return dot;
  }

  /**
     @brief GenMatrix addition operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int M, int N> __device__ __host__ inline
    GenMatrix<Float,M,N> operator+(const GenMatrix<Float,M,N> &x, const GenMatrix<Float,M,N> &y) {

    GenMatrix<Float,M,N> z;

#pragma unroll
    for (int r=0; r<N; r++) {
#pragma unroll
      for (int c=0; c<N; c++) {
        z.data[c*M + r] = x.data[c*M + r] + y.data[c*M + r];
      }
    }

    return z;
  }

  /**
     @brief GenMatrix subtraction operator
     @param[in] x Input vector
     @param[in] y Input vector
     @return The vector x + y
  */
  template<typename Float, int M, int N> __device__ __host__ inline
    GenMatrix<Float,M,N> operator-(const GenMatrix<Float,M,N> &x, const GenMatrix<Float,M,N> &y) {

    GenMatrix<Float,M,N> z;

#pragma unroll
    for (int r=0; r<M; r++) {
#pragma unroll
      for (int c=0; c<N; c++) {
        z.data[c*M + r] = x.data[c*M + r] - y.data[c*M + r];
      }
    }

    return z;
  }

  /**
     @brief Compute the scalar-vector product y = a * x
     @param[in] a Input scalar
     @param[in] x Input vector
     @return The vector a * x
  */
  template<typename Float, int M, int N, typename S> __device__ __host__ inline
    GenMatrix<Float,M,N> operator*(const S &a, const GenMatrix<Float,M,N> &x) {

    GenMatrix<Float,M,N> y;

#pragma unroll
    for (int r=0; r<M; r++) {
#pragma unroll
      for (int c=0; c<N; c++) {
        y.data[c*M + r] = a * x.data[c*M + r];
      }
    }

    return y;
  }

} // namespace quda
