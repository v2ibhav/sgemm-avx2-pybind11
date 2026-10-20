import time, numpy as np, vaibhav_math

n = 1024
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)
fl = 2 * n ** 3

for name, f in [("engine", lambda: vaibhav_math.multiply(A, B)), ("numpy ", lambda: A @ B)]:
    for _ in range(30): f()                       # warm-up
    t = []
    for _ in range(200):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    t = np.array(t)
    print(f"{name}: best {fl/t.min()/1e9:.0f} | median {fl/np.median(t)/1e9:.0f} | worst {fl/t.max()/1e9:.0f} GFLOPS")