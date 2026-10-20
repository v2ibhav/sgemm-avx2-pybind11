#include <algorithm>
#include <immintrin.h> 
#include <omp.h>       
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // NEW: NumPy direct support

namespace py = pybind11;
using namespace std;

const int BLOCK_SIZE = 64;

// Ab function std::vector ki jagah direct NumPy arrays (py::array_t) accept karega
py::array_t<float> multiply_matrices(py::array_t<float> input_A, py::array_t<float> input_B, int M, int N, int K) {
    
    // 1. Direct Memory Access (Zero-Copy Magic)
    // Hum arrays ko copy nahi kar rahe, seedha unka memory address (pointer) maang rahe hain
    py::buffer_info bufA = input_A.request();
    py::buffer_info bufB = input_B.request();

    float *A = static_cast<float *>(bufA.ptr);
    float *B = static_cast<float *>(bufB.ptr);

    // 2. Output ke liye naya NumPy array direct C++ mein banayenge
    auto result = py::array_t<float>(M * K);
    py::buffer_info bufC = result.request();
    float *C = static_cast<float *>(bufC.ptr);

    // Output array ko 0 se initialize karna
    #pragma omp parallel for
    for (int idx = 0; idx < M * K; ++idx) {
        C[idx] = 0.0f;
    }

    // 3. The Core Engine (Same SIMD Math logic, par ab direct memory pointers ke sath)
    #pragma omp parallel for
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < K; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                
                for (int i = ii; i < min(ii + BLOCK_SIZE, M); ++i) {
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); ++k) {
                        
                        __m256 a_val = _mm256_set1_ps(A[i * N + k]);
                        
                        int j = jj;
                        int end_j = min(jj + BLOCK_SIZE, K);
                        
                        for (; j <= end_j - 8; j += 8) {
                            __m256 c_vals = _mm256_loadu_ps(&C[i * K + j]);
                            __m256 b_vals = _mm256_loadu_ps(&B[k * K + j]);
                            
                            __m256 result_vec = _mm256_add_ps(c_vals, _mm256_mul_ps(a_val, b_vals));
                            _mm256_storeu_ps(&C[i * K + j], result_vec);
                        }
                        
                        for (; j < end_j; ++j) {
                            C[i * K + j] += A[i * N + k] * B[k * K + j];
                        }
                    }
                }
            }
        }
    }
    return result; // Direct NumPy array wapas Python ko de rahe hain
}

PYBIND11_MODULE(vaibhav_math, m) {
    m.doc() = "Vaibhav's Zero-Copy Hardware-Optimized Math Library";
    m.def("multiply", &multiply_matrices, "Multiplies NumPy arrays directly with Zero-Copy overhead");
}
