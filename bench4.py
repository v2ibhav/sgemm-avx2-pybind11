import time, numpy as np, vaibhav_math

for n in [2032, 2040, 2048, 2056, 2064]:
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    vaibhav_math.multiply(A, B)
    best = min((lambda s: (vaibhav_math.multiply(A, B), time.perf_counter() - s)[1])(time.perf_counter()) for _ in range(10))
    print(n, f"{2*n**3/best/1e9:.0f} GFLOPS")