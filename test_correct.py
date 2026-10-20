import numpy as np
import vaibhav_math

rng = np.random.default_rng(0)
ok = True

def check(name, A, B):
    global ok
    got = vaibhav_math.multiply(A, B)
    want = np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32)
    good = got.shape == want.shape and np.allclose(got, want, rtol=1e-3, atol=1e-3)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}")

# 1. Odd and awkward shapes (M, N, K)
for M, N, K in [(1, 1, 1), (7, 5, 3), (100, 37, 53), (64, 64, 64),
                (65, 129, 63), (3, 200, 9), (257, 33, 130)]:
    A = rng.random((M, N), dtype=np.float32)
    B = rng.random((N, K), dtype=np.float32)
    check(f"shape {M}x{N} @ {N}x{K}", A, B)

# 2. Non-contiguous input (transpose and slice) -- used to give silently wrong results
A = rng.random((50, 80), dtype=np.float32)
B = rng.random((60, 80), dtype=np.float32)
check("transposed B (non-contiguous)", A, B.T)
check("sliced A (non-contiguous)", A[::2, :], B.T)

# 3. float64 input is auto-converted to float32
check("float64 input", A.astype(np.float64), B.T.astype(np.float64))

# 4. Wrong shapes must raise an error, not crash
try:
    vaibhav_math.multiply(rng.random((4, 5), dtype=np.float32),
                          rng.random((6, 3), dtype=np.float32))
    print("FAIL  shape mismatch did not raise")
    ok = False
except ValueError:
    print("PASS  shape mismatch raises ValueError")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
