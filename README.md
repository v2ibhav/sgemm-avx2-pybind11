# Optimized SGEMM Kernel in C++ (AVX2/FMA + OpenMP) with Python Bindings

A single-precision (float32) matrix multiplication kernel written from scratch in C++17 and exposed to
Python through Pybind11. Built as a learning project to understand how real BLAS libraries get their
speed: cache blocking, SIMD, register blocking, data packing, and multi-threading.

It is **not** a replacement for OpenBLAS/MKL. The goal was to measure how far each optimization
step goes and to understand why.

## Usage

```python
import numpy as np, vaibhav_math

A = np.random.rand(1024, 2048).astype(np.float32)
B = np.random.rand(2048, 1024).astype(np.float32)
C = vaibhav_math.multiply(A, B)          # shapes are read from the arrays, C is (1024, 1024)
```

Inputs are made C-contiguous and converted to float32 automatically; mismatched shapes raise `ValueError`.
Input arrays are read in place (no copy when already contiguous float32).

## Build

```bash
python3 -m venv ~/bench_env && source ~/bench_env/bin/activate
pip install numpy pybind11 threadpoolctl
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) fast_math.cpp \
    -o vaibhav_math$(python3-config --extension-suffix) -mavx2 -mfma -fopenmp
python3 test_correct.py     # correctness: odd shapes, non-contiguous input, dtype, error handling
python3 bench5.py           # size sweep against NumPy
```

Requires a CPU with AVX2 and FMA.

## What the kernel does

1. **Cache blocking:** C is split into MC x NC blocks, the k dimension into KC chunks (96 / 256 / 256).
2. **6x16 register-blocked microkernel:** a 6x16 tile of C lives in 12 AVX2 registers for the whole k loop,
   so each k step costs 2 vector loads of B, 6 broadcasts of A and 12 FMAs, with no stores to C.
3. **Packing of B:** each B tile is copied into a contiguous per-thread buffer (16-column strips) so the
   microkernel reads memory sequentially.
4. **OpenMP:** `collapse(2)` over C blocks with dynamic scheduling. Each C block is owned by one thread, so there are no data races.
5. **Edge handling:** sizes not divisible by 6 or 16 fall back to scalar code.

## Results

**Test machine:** Intel Core i5-12500H (4 P-cores + 8 E-cores, 16 threads), WSL2 Ubuntu, GCC,
Python 3.14, NumPy with OpenBLAS (pip, virtual environment), float32.

GFLOPS = 2*M*N*K / time. Benchmarks are noisy on this laptop, so ranges across several runs are shown.

### Ablation: 1024 x 2048 x 1024 (20 trials per run)

| Version | Change | Median GFLOPS | Best GFLOPS |
|---|---|---|---|
| v0 | tiling + AVX2 + OpenMP | ~80-86 | ~94 |
| v1 | + FMA, `collapse(2)` | ~91-105 | ~102-110 |
| v2 | + 6x16 register-blocked microkernel | ~288-408 | ~386-428 |
| v3 | + packing of B | ~285-491 | ~361-530 |

At this particular shape v3 is not clearly better than v2; the benefit of packing shows at power-of-two sizes (below).

### Size sweep, square matrices (best of 8, GFLOPS)

| n | v2 | v3 | OpenBLAS (same sessions) |
|---|---|---|---|
| 2048 | ~335-359 | ~400-510 | ~330-460 |
| 4096 | ~296 (one run) | ~455-486 | ~445-463 |
| 3072 | ~405-430 | ~445-535 | ~437-459 |
| 2032 / 2056 | ~510-560 | ~410-550 | ~357-467 |

For n >= 2032 the kernel is roughly on par with OpenBLAS on this machine (within about +-15%), but
OpenBLAS's own numbers moved between ~290 and ~475 GFLOPS across sessions, so single ratios are not reliable.
At n = 1024 both engines were unstable (engine 210-481 GFLOPS) and I do not claim a figure there.

### The 2048 dip

Before packing, n = 2048 ran ~35% slower than n = 2032 or 2056 (reproduced in 4 runs). The pattern is
consistent with cache set conflicts: with a power-of-two row stride, the rows of a B strip map to a small
number of cache sets. Packing B contiguously removed the dip. I did not confirm the mechanism with hardware
counters (`perf` is not usable under my WSL2 setup), so this is the most likely explanation, not a proven one.

## Limitations

- Measured on **one machine** (hybrid CPU, WSL2). Block sizes (MC/NC/KC) were not auto-tuned.
- float32 only, CPU only, no `alpha`/`beta`, no transposed-operand flags (inputs are copied to contiguous layout).
- Edge tiles use plain scalar code, so awkward sizes (not multiples of 6 and 16) lose some speed.
- Timing noise is large; compare ranges, not single numbers.
- Not a drop-in BLAS replacement and not tested for numerical edge cases (NaN/Inf, tiny/huge values).

## Files

| File | Purpose |
|---|---|
| `fast_math.cpp` | current kernel (v3) |
| `fast_math_v0.cpp` ... `fast_math_v2.cpp` | earlier versions, kept for the ablation |
| `test_correct.py` | correctness tests |
| `bench2.py`, `bench5.py` | benchmarks |

## What I learned

- Kernel speed was limited by load/store traffic, not arithmetic: FMA alone gave ~10-15%, the register-blocked microkernel gave ~3-4x.
- A benchmark is only as good as its baseline: my first "5x faster than NumPy" result came from NumPy using the reference (unoptimized) BLAS. After switching to OpenBLAS the real gap appeared.
