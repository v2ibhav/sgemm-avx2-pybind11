import time
import numpy as np
import vaibhav_math


def best_time(f, n=8):
    f()  # warm-up
    t = []
    for _ in range(n):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return min(t)


_w = np.random.rand(2048, 2048).astype(np.float32)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2:   # 2 second ka warm-up
    vaibhav_math.multiply(_w, _w); _w @ _w


print(f"{'n':>6} | {'engine':>7} | {'numpy':>7} | ratio   (square n x n x n, best of 8, GFLOPS)")
for n in [1024, 2032, 2048, 2056, 3072, 4096]:
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    ok = np.allclose(vaibhav_math.multiply(A, B), A @ B, rtol=1e-3, atol=1e-3)
    fl = 2 * n ** 3
    e = fl / best_time(lambda: vaibhav_math.multiply(A, B)) / 1e9
    p = fl / best_time(lambda: A @ B) / 1e9
    print(f"{n:>6} | {e:7.0f} | {p:7.0f} | {100 * e / p:4.0f}%  {'OK' if ok else 'WRONG RESULT'}")
