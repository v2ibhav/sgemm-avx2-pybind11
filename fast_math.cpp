// fast_math_v3.cpp -- v3: v2 microkernel + packing of B
//
// Problem in v2: the microkernel reads a 16-column strip of B, one row per k step.
// Rows of B are K*4 bytes apart. When K is a power of two (2048 -> 8 KB, 4096 -> 16 KB)
// all those 64-byte pieces land in the same few cache sets and evict each other
// (cache aliasing), so the kernel keeps re-fetching data.
//
// Fix: before using a (KC x NC) tile of B, copy it into a small per-thread buffer where
// each 16-column strip is stored contiguously (k-major, 16 floats per k). The microkernel
// then reads this buffer strictly sequentially, whatever K is.
//
// Compile (one line, inside your venv; same module name so tests/bench work unchanged):
// g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) fast_math_v3.cpp -o vaibhav_math$(python3-config --extension-suffix) -mavx2 -mfma -fopenmp

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr int MR = 6;
constexpr int NR = 16;
constexpr int MC = 96;   // multiple of MR
constexpr int NC = 256;  // multiple of NR
constexpr int KC = 256;

using Mat = py::array_t<float, py::array::c_style | py::array::forcecast>;

// C[6x16] += A[6 x kc] * Bp[kc x 16]   (Bp row stride = ldb floats; 16 when packed)
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

// Scalar code for leftover edges, reads the ORIGINAL (unpacked) A and B.
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

    #pragma omp parallel
    {
        // Each thread owns its own packing buffer (KC x NC floats = 256 KB)
        std::vector<float> Bp((std::size_t)KC * NC);

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int ii = 0; ii < M; ii += MC) {
            for (int jj = 0; jj < K; jj += NC) {
                const int i_end = std::min(ii + MC, M);
                const int j_end = std::min(jj + NC, K);
                const int nstrips = (j_end - jj) / NR;      // full 16-wide strips
                const int j_full = jj + nstrips * NR;       // first column NOT packed

                for (int kk = 0; kk < N; kk += KC) {
                    const int kc = std::min(KC, N - kk);

                    // ---- pack B[kk..kk+kc, jj..j_full) into strip-major layout ----
                    for (int s = 0; s < nstrips; ++s) {
                        float *dst = Bp.data() + (std::size_t)s * kc * NR;
                        const float *src = B + (std::size_t)kk * K + jj + s * NR;
                        for (int k = 0; k < kc; ++k) {
                            const float *row = src + (std::size_t)k * K;
                            _mm256_storeu_ps(dst + k * NR,     _mm256_loadu_ps(row));
                            _mm256_storeu_ps(dst + k * NR + 8, _mm256_loadu_ps(row + 8));
                        }
                    }

                    int i = ii;
                    for (; i + MR <= i_end; i += MR) {
                        for (int s = 0; s < nstrips; ++s)
                            kernel_6x16(A + (std::size_t)i * N + kk,
                                        Bp.data() + (std::size_t)s * kc * NR,
                                        C + (std::size_t)i * K + jj + s * NR,
                                        kc, N, NR, K);
                        if (j_full < j_end)  // leftover columns for these 6 rows
                            edge(A, B, C, N, K, i, i + MR, j_full, j_end, kk, kk + kc);
                    }
                    if (i < i_end)           // leftover rows
                        edge(A, B, C, N, K, i, i_end, jj, j_end, kk, kk + kc);
                }
            }
        }
    }
    return result;
}

PYBIND11_MODULE(vaibhav_math, m) {
    m.doc() = "AVX2/FMA register-blocked SGEMM with B packing (float32), NumPy bindings";
    m.def("multiply", &multiply, py::arg("A"), py::arg("B"),
          "C = A @ B for 2D float32 arrays. Shapes are read from the arrays.");
}
