import time
import numpy as np
import vaibhav_math

M, N, K = 1024, 2048, 1024
TRIALS = 20

A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)
flops = 2 * M * N * K


def bench(f):
    f()  # warm-up (not counted)
    times = []
    for _ in range(TRIALS):
        s = time.perf_counter()
        f()
        times.append(time.perf_counter() - s)
    return float(np.median(times)), min(times)


# Engine and NumPy run in SEPARATE loops so their threads don't disturb each other
results = {}
for name, f in [("Engine", lambda: vaibhav_math.multiply(A, B)),
                ("NumPy ", lambda: A @ B)]:
    med, best = bench(f)
    results[name] = flops / med / 1e9
    print(f"{name}: median {med:.4f}s ({flops/med/1e9:.1f} GFLOPS) | "
          f"best {best:.4f}s ({flops/best/1e9:.1f} GFLOPS)")

print(f"\nEngine = {100 * results['Engine'] / results['NumPy ']:.1f}% of NumPy (median)")