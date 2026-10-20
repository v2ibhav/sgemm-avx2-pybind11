import time
import numpy as np
import vaibhav_math

def bench(f, trials=15):
    f()
    t = []
    for _ in range(trials):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return float(np.median(t)), min(t)

print(f"{'size (MxNxK)':>18} | {'ok':>3} | {'engine med/best':>17} | {'numpy med/best':>17} | best ratio")
for M, N, K in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
                (3072, 3072, 3072), (1000, 1500, 777)]:
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(N, K).astype(np.float32)
    ok = np.allclose(vaibhav_math.multiply(A, B), A @ B, rtol=1e-3, atol=1e-3)
    fl = 2 * M * N * K
    em, eb = bench(lambda: vaibhav_math.multiply(A, B))
    nm, nb = bench(lambda: A @ B)
    print(f"{M:>5}x{N:>5}x{K:>5} | {'OK' if ok else 'BAD':>3} | "
          f"{fl/em/1e9:7.0f}/{fl/eb/1e9:<7.0f} | {fl/nm/1e9:7.0f}/{fl/nb/1e9:<7.0f} | {100*(fl/eb)/(fl/nb):5.0f}%")