import numpy as np

from core.tensor.tensor import Tensor


def check(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def check_close(actual: np.ndarray, expected: np.ndarray, msg: str) -> None:
    if not np.allclose(actual, expected, atol=1e-8, rtol=1e-8):
        raise AssertionError(f"{msg} | actual={actual} expected={expected}")


def run() -> None:
    # 1) add/mul chain: z = x*y + x
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = (x * y) + x
    z.backward()
    check_close(x.grad, np.array(4.0), "dz/dx should be y + 1 = 4")
    check_close(y.grad, np.array(2.0), "dz/dy should be x = 2")

    # 2) accumulation through reused tensor: z = x*x + x
    x2 = Tensor(3.0, requires_grad=True)
    z2 = (x2 * x2) + x2
    z2.backward()
    check_close(x2.grad, np.array(7.0), "d(x*x + x)/dx should be 2x + 1 = 7")

    # 3) reverse subtraction: z = 10 - x
    x3 = Tensor(5.0, requires_grad=True)
    z3 = 10.0 - x3
    z3.backward()
    check_close(x3.grad, np.array(-1.0), "d(10-x)/dx should be -1")

    # 4) pow: z = x^2
    x4 = Tensor(4.0, requires_grad=True)
    z4 = x4.pow(2)
    z4.backward()
    check_close(x4.grad, np.array(8.0), "d(x^2)/dx at x=4 should be 8")

    # 5) sum over vector
    v = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = v.sum()
    s.backward()
    check_close(v.grad, np.array([1.0, 1.0, 1.0]), "d(sum(v))/dv should be ones")

    # 6) mean over vector
    v2 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    m = v2.mean()
    m.backward()
    check_close(v2.grad, np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), "d(mean(v))/dv should be 1/n")


if __name__ == "__main__":
    run()
    print("check_backward_basic: PASS")
