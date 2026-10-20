// fast_math_v2.cpp -- v2: 6x16 register-blocked microkernel (no packing yet)
//
// Idea: keep a 6x16 tile of C in 12 AVX2 registers for the WHOLE k loop.
// Per k step we only do: 2 loads of B (16 floats), 6 broadcasts of A,
// 12 FMAs. C is loaded once at the start and stored once at the end.
// v1 did 2 loads + 1 store for EVERY single FMA.
//
// Registers: 12 (C) + 2 (B) + 1 (A broadcast) = 15 of 16 ymm registers.
//
// Compile (one line, inside your venv, keep the same module name so the tests work):
// g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) fast_math_v2.cpp -o vaibhav_math$(python3-config --extension-suffix) -mavx2 -mfma -fopenmp

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Microkernel tile size
constexpr int MR = 6;   // rows of C kept in registers
constexpr int NR = 16;  // columns of C kept in registers (2 x __m256)

// Cache blocking sizes (tune these later)
constexpr int MC = 96;   // multiple of MR
constexpr int NC = 256;  // multiple of NR
constexpr int KC = 256;

using Mat = py::array_t<float, py::array::c_style | py::array::forcecast>;

// C[6x16] += A[6 x kc] * B[kc x 16]
// lda/ldb/ldc are the row strides (in floats) of A, B, C.
static inline void kernel_6x16(const float *A, const float *B, float *C,
                               int kc, int lda, int ldb, int ldc) {
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc), c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc), c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc), c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc), c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc), c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc), c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    for (int k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(B + (std::size_t)k * ldb);
        const __m256 b1 = _mm256_loadu_ps(B + (std::size_t)k * ldb + 8);
        __m256 a;

        a = _mm256_broadcast_ss(A + 0 * lda + k);
        c00 = _mm256_fmadd_ps(a, b0, c00);  c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1 * lda + k);
        c10 = _mm256_fmadd_ps(a, b0, c10);  c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2 * lda + k);
        c20 = _mm256_fmadd_ps(a, b0, c20);  c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3 * lda + k);
        c30 = _mm256_fmadd_ps(a, b0, c30);  c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4 * lda + k);
        c40 = _mm256_fmadd_ps(a, b0, c40);  c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5 * lda + k);
        c50 = _mm256_fmadd_ps(a, b0, c50);  c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C + 0 * ldc, c00);  _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10);  _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20);  _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30);  _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40);  _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50);  _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

// Plain scalar code for the leftover edges (sizes not divisible by 6 or 16).
static void edge(const float *A, const float *B, float *C, int N, int K,
                 int i0, int i1, int j0, int j1, int k0, int k1) {
    for (int i = i0; i < i1; ++i)
        for (int k = k0; k < k1; ++k) {
            const float a = A[(std::size_t)i * N + k];
            for (int j = j0; j < j1; ++j)
                C[(std::size_t)i * K + j] += a * B[(std::size_t)k * K + j];
        }
}

py::array_t<float> multiply(Mat A_arr, Mat B_arr) {
    if (A_arr.ndim() != 2 || B_arr.ndim() != 2)
        throw std::invalid_argument("A and B must be 2D arrays");
    if (A_arr.shape(1) != B_arr.shape(0))
        throw std::invalid_argument("Shape mismatch: A is (M,N) so B must be (N,K)");

    const int M = static_cast<int>(A_arr.shape(0));
    const int N = static_cast<int>(A_arr.shape(1));
    const int K = static_cast<int>(B_arr.shape(1));

    const float *A = A_arr.data();
    const float *B = B_arr.data();
    py::array_t<float> result({(py::ssize_t)M, (py::ssize_t)K});
    float *C = result.mutable_data();

    py::gil_scoped_release release;

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)M * K; ++idx) C[idx] = 0.0f;

    // Each (ii, jj) block of C belongs to exactly one thread -> no race condition.
    // dynamic schedule: fast P-cores simply take more blocks than slow E-cores.
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int ii = 0; ii < M; ii += MC) {
        for (int jj = 0; jj < K; jj += NC) {
            const int i_end = std::min(ii + MC, M);
            const int j_end = std::min(jj + NC, K);

            for (int kk = 0; kk < N; kk += KC) {
                const int kc = std::min(KC, N - kk);

                int i = ii;
                for (; i + MR <= i_end; i += MR) {
                    int j = jj;
                    for (; j + NR <= j_end; j += NR)
                        kernel_6x16(A + (std::size_t)i * N + kk,
                                    B + (std::size_t)kk * K + j,
                                    C + (std::size_t)i * K + j,
                                    kc, N, K, K);
                    if (j < j_end)  // leftover columns for these 6 rows
                        edge(A, B, C, N, K, i, i + MR, j, j_end, kk, kk + kc);
                }
                if (i < i_end)      // leftover rows
                    edge(A, B, C, N, K, i, i_end, jj, j_end, kk, kk + kc);
            }
        }
    }
    return result;
}

PYBIND11_MODULE(vaibhav_math, m) {
    m.doc() = "AVX2/FMA register-blocked SGEMM (float32) with NumPy bindings";
    m.def("multiply", &multiply, py::arg("A"), py::arg("B"),
          "C = A @ B for 2D float32 arrays. Shapes are read from the arrays.");
}
