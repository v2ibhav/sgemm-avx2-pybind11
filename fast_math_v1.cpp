
// fast_math.cpp  --  v1 (Steps 1-4)
// Changes vs v0:
//   1. Shape + contiguity checks, M/N/K derived from the arrays, 2D output
//   2. FMA (_mm256_fmadd_ps) instead of separate mul + add
//   3. OpenMP collapse(2) over (ii, jj) tiles -> more parallel work, still race-free
//   4. Odd sizes handled (scalar tail) -> test with test_correct.py
//
// Compile (inside your venv):
//   g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) 
//       fast_math.cpp -o vaibhav_math$(python3-config --extension-suffix) -mavx2 -mfma -fopenmp

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr int BLOCK = 64;

// c_style  -> array is forced to be contiguous (row-major), so A.T or slices are
//             copied into a proper layout instead of silently giving wrong results.
// forcecast-> float64/int arrays are converted to float32 automatically.
using Mat = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> multiply(Mat A_arr, Mat B_arr) {
    // ---- Step 1: validation ----
    if (A_arr.ndim() != 2 || B_arr.ndim() != 2)
        throw std::invalid_argument("A and B must be 2D arrays");
    if (A_arr.shape(1) != B_arr.shape(0))
        throw std::invalid_argument("Shape mismatch: A is (M,N) so B must be (N,K)");

    const int M = static_cast<int>(A_arr.shape(0));
    const int N = static_cast<int>(A_arr.shape(1));
    const int K = static_cast<int>(B_arr.shape(1));

    const float *A = A_arr.data();
    const float *B = B_arr.data();

    // 2D output (M x K) instead of a flat 1D array
    py::array_t<float> result({static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(K)});
    float *C = result.mutable_data();

    // Release the Python GIL while the heavy C++ work runs
    py::gil_scoped_release release;

    // C must start at 0 because we accumulate into it
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < static_cast<long long>(M) * K; ++idx)
        C[idx] = 0.0f;

    // ---- Step 3: collapse(2) over C tiles. Each (ii, jj) tile of C is owned by
    // exactly one thread, so there is no race condition. ----
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BLOCK) {
        for (int jj = 0; jj < K; jj += BLOCK) {
            const int i_end = std::min(ii + BLOCK, M);
            const int j_end = std::min(jj + BLOCK, K);

            for (int kk = 0; kk < N; kk += BLOCK) {
                const int k_end = std::min(kk + BLOCK, N);

                for (int i = ii; i < i_end; ++i) {
                    float *Ci = C + static_cast<std::size_t>(i) * K;
                    const float *Ai = A + static_cast<std::size_t>(i) * N;

                    for (int k = kk; k < k_end; ++k) {
                        const float a = Ai[k];
                        const __m256 a_val = _mm256_set1_ps(a);
                        const float *Bk = B + static_cast<std::size_t>(k) * K;

                        int j = jj;
                        // ---- Step 2: FMA: c = a*b + c in ONE instruction ----
                        for (; j <= j_end - 8; j += 8) {
                            __m256 c_vals = _mm256_loadu_ps(Ci + j);
                            __m256 b_vals = _mm256_loadu_ps(Bk + j);
                            c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                            _mm256_storeu_ps(Ci + j, c_vals);
                        }
                        // ---- Step 4: scalar tail for sizes not divisible by 8 ----
                        for (; j < j_end; ++j)
                            Ci[j] += a * Bk[j];
                    }
                }
            }
        }
    }
    return result;
}

PYBIND11_MODULE(vaibhav_math, m) {
    m.doc() = "AVX2/FMA + OpenMP tiled SGEMM (float32) with NumPy bindings";
    m.def("multiply", &multiply, py::arg("A"), py::arg("B"),
          "C = A @ B for 2D float32 arrays. Shapes are read from the arrays.");
}
